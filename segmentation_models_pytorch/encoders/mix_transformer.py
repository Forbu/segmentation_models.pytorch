# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math
import torch
import torch.nn as nn
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_angle_flatten(dimension_hw):
    """
    Calculate polar angles for each pixel relative to the image center.

    Args:
        dimension_hw (int): Size of the square image (either 104, 52, 26, or 13)

    Returns:
        torch.Tensor: Flattened tensor of angles in radians relative to the image center
    """
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(dimension_hw), torch.arange(dimension_hw), indexing="ij"
    )

    # Calculate center coordinates
    center_y = (dimension_hw - 1) / 2
    center_x = (dimension_hw - 1) / 2

    # Calculate relative coordinates from center
    y_from_center = y - center_y
    x_from_center = x - center_x

    # Calculate angles using arctan2
    angles = torch.atan2(y_from_center, x_from_center)

    # Flatten the angles tensor
    angles_flat = angles.flatten()

    return angles_flat


def get_rotation_matrices(angle, freq_array):
    """
    Get rotation matrices for a given angle and frequency array

    Args:
        angle (torch.Tensor): Angle tensor size (N)
        freq_array (torch.Tensor): Frequency array (size dim)

    Returns:
        torch.Tensor: Rotation matrices size (N, size dim)
    """

    rotation_first = torch.cos(angle[:, None] * freq_array[None, :])
    rotation_second = torch.sin(angle[:, None] * freq_array[None, :])

    return rotation_first, rotation_second


def compute_rotation_embedding(q, rotation_first, rotation_second):
    """
    Compute rotation embedding for q

    We want to rotate q by the rotation matrices

    q_embed = q * rotation_first + q_rot * rotation_second

    if q = [q0, q1, q2, q3 etc] then q_rot = [q1, -q0, q3, -q2, etc]

    Args:
        q (torch.Tensor): Query tensor size (B, N, C)
        rotation_first (torch.Tensor): Rotation first tensor size (N, C)
        rotation_second (torch.Tensor): Rotation second tensor size (N, C)

    Returns:
        torch.Tensor: Rotated query tensor size (B, N, C)
    """

    q_rot = torch.stack([q[..., 1::2], -q[..., ::2]], dim=-1).flatten(-2)
    q_embed = (
        q * rotation_first.unsqueeze(0).unsqueeze(0)
        + q_rot * rotation_second.unsqueeze(0).unsqueeze(0)
    )

    return q_embed


def get_freq_array(dim):
    """
    Get frequency array for a given dimension
    """

    # we only want to rotate half the features
    index = torch.arange(dim // 2)

    # frequency array
    freq = torch.exp(index * -math.log(10000) / (dim // 2 - 1))

    # interleave the frequency array to get the full frequency array
    # [a, b] -> [a, a, b, b]
    freq = torch.repeat_interleave(freq, 2, dim=-1)

    return freq


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # Initialize caching attributes as None
        self.register_buffer("angle_q", None)
        self.register_buffer("angle_k", None)
        self.register_buffer("freq_array", None)
        self.register_buffer("rotation_first_q", None)
        self.register_buffer("rotation_second_q", None)
        self.register_buffer("rotation_first_k", None)
        self.register_buffer("rotation_second_k", None)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_)

            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)

            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        # Calculate angles if not already computed
        if self.angle_q is None:
            self.angle_q = get_angle_flatten(H)

            # convert to device of q
            self.angle_q = self.angle_q.to(q.device)

        if self.angle_k is None:
            k_size = int(math.sqrt(k.shape[2]))
            self.angle_k = get_angle_flatten(k_size)

            # convert to device of k
            self.angle_k = self.angle_k.to(k.device)

        # Get frequency array if not already computed
        if self.freq_array is None:
            self.freq_array = get_freq_array(C // self.num_heads)

            # convert to device of q
            self.freq_array = self.freq_array.to(q.device)

        # Get rotation matrices if not already computed
        if self.rotation_first_q is None or self.rotation_second_q is None:
            self.rotation_first_q, self.rotation_second_q = get_rotation_matrices(
                self.angle_q, self.freq_array
            )

            # convert to device of q
            self.rotation_first_q = self.rotation_first_q.to(q.device)
            self.rotation_second_q = self.rotation_second_q.to(q.device)

        if self.rotation_first_k is None or self.rotation_second_k is None:
            self.rotation_first_k, self.rotation_second_k = get_rotation_matrices(
                self.angle_k, self.freq_array
            )

            # convert to device of k
            self.rotation_first_k = self.rotation_first_k.to(k.device)
            self.rotation_second_k = self.rotation_second_k.to(k.device)

        # Apply rotation angle embedding using cached matrices
        q = compute_rotation_embedding(q, self.rotation_first_q, self.rotation_second_q)
        k = compute_rotation_embedding(k, self.rotation_first_k, self.rotation_second_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        context_dim=3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.context_dim = context_dim
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # simple seqeuntial element for context enmbedding
        # TODO
        self.film_layer = nn.Sequential(
            nn.Linear(self.context_dim, 32), nn.GELU(), nn.Linear(32, embed_dims[0] * 2)
        )

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, context=None):
        """
        x is of size (Batch, Channel, W, H)
        and context is (Batch, Context(3))
        """
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)

        if context != None:
            context_embedding = self.film_layer(context).unsqueeze(1)
            x = (
                x * context_embedding[:, :, : self.embed_dims[0]]
                + context_embedding[:, :, -self.embed_dims[0] :]
            )

        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)

        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)

        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)

        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, context=None):
        x = self.forward_features(x, context)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# ---------------------------------------------------------------
# End of NVIDIA code
# ---------------------------------------------------------------

from ._base import EncoderMixin  # noqa E402


class MixVisionTransformerEncoder(MixVisionTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

    def make_dilated(self, *args, **kwargs):
        raise ValueError("MixVisionTransformer encoder does not support dilated mode")

    def set_in_channels(self, in_channels, *args, **kwargs):
        if in_channels != 3:
            raise ValueError(
                "MixVisionTransformer encoder does not support in_channels setting other than 3"
            )

    def forward(self, x):
        # create dummy output for the first block
        B, C, H, W = x.shape
        dummy = torch.empty([B, 0, H // 2, W // 2], dtype=x.dtype, device=x.device)

        return [x, dummy] + self.forward_features(x)[: self._depth - 1]

    def load_state_dict(self, state_dict):
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        return super().load_state_dict(state_dict)


class MixRadioTransformerEncoder(MixVisionTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 14

    def make_dilated(self, *args, **kwargs):
        raise ValueError("MixVisionTransformer encoder does not support dilated mode")

    def set_in_channels(self, in_channels, *args, **kwargs):
        if in_channels != 14:
            raise ValueError(
                "MixVisionTransformer encoder does not support in_channels setting other than 14"
            )

    def forward(self, x, context=None):
        # create dummy output for the first block
        B, C, H, W = x.shape
        dummy = torch.empty([B, 0, H // 2, W // 2], dtype=x.dtype, device=x.device)

        if context is None:
            return [x, dummy] + self.forward_features(x)[: self._depth - 1]
        else:
            return [x, dummy] + self.forward_features(x, context)[: self._depth - 1]


def get_pretrained_cfg(name):
    return {
        "url": "https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/{}.pth".format(
            name
        ),
        "input_space": "RGB",
        "input_size": [3, 224, 224],
        "input_range": [0, 1],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


mix_transformer_encoders = {
    "mit_radio": {
        "encoder": MixRadioTransformerEncoder,
        "params": dict(
            context_dim=3,
            in_chans=14,
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b0": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {"imagenet": get_pretrained_cfg("mit_b0")},
        "params": dict(
            out_channels=(3, 0, 32, 64, 160, 256),
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b1": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {"imagenet": get_pretrained_cfg("mit_b1")},
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b2": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {"imagenet": get_pretrained_cfg("mit_b2")},
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b3": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {"imagenet": get_pretrained_cfg("mit_b3")},
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b4": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {"imagenet": get_pretrained_cfg("mit_b4")},
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b5": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {"imagenet": get_pretrained_cfg("mit_b5")},
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
}
