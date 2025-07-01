# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from .modules import FeatureFusion as FFM
from .modules import FeatureCorrection_s2c as FCM
from mmrgbx.registry import MODELS

layer_scale = False
init_value = 1e-6


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
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
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
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
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
    ):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SABlock(nn.Module):
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
    ):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
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

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SABlock_Windows(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=14,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
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

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(
            x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2).reshape(B, C, H, W)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


@MODELS.register_module()
class DualUniFormer(nn.Module):
    ARCH = {
        "small": dict(
            layers=[3, 4, 8, 3],
            embed_dim=[64, 128, 320, 512],
            head_dim=64,
            mlp_ratio=4,
            qkv_bias=True,
        ),
        "smallp32": dict(
            layers=[3, 5, 9, 3],
            embed_dim=[64, 128, 320, 512],
            head_dim=32,
            mlp_ratio=4,
            qkv_bias=True,
        ),
        "smallp64": dict(
            layers=[3, 5, 9, 3],
            embed_dim=[64, 128, 320, 512],
            head_dim=64,
            mlp_ratio=4,
            qkv_bias=True,
        ),
        "base": dict(
            layers=[5, 8, 20, 7],
            embed_dim=[64, 128, 320, 512],
            head_dim=64,
            mlp_ratio=4,
            qkv_bias=True,
        ),
        "large": dict(
            layers=[5, 8, 20, 7],
            embed_dim=[64, 128, 320, 512],
            head_dim=64,
            mlp_ratio=4,
            qkv_bias=True,
        ),
    }

    def __init__(
        self,
        arch,
        img_size=224,
        in_chans=3,
        qk_scale=None,
        norm_fuse=nn.BatchNorm2d,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        windows=False,
        hybrid=False,
        window_size=14,
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()

        arch = self.ARCH[arch]
        layers = arch["layers"]
        embed_dim = arch["embed_dim"]
        head_dim = arch["head_dim"]
        mlp_ratio = arch["mlp_ratio"]
        qkv_bias = arch["qkv_bias"]

        self.windows = windows
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0]
        )
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=2,
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1],
        )
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=2,
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2],
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16,
            patch_size=2,
            in_chans=embed_dim[2],
            embed_dim=embed_dim[3],
        )

        self.aux_patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0]
        )
        self.aux_patch_embed2 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=2,
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1],
        )
        self.aux_patch_embed3 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=2,
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2],
        )
        self.aux_patch_embed4 = PatchEmbed(
            img_size=img_size // 16,
            patch_size=2,
            in_chans=embed_dim[2],
            embed_dim=embed_dim[3],
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.aux_pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(layers[0])
            ]
        )
        self.norm1 = norm_layer(embed_dim[0])

        self.aux_blocks1 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(layers[0])
            ]
        )
        self.aux_norm1 = norm_layer(embed_dim[0])

        self.blocks2 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + layers[0]],
                    norm_layer=norm_layer,
                )
                for i in range(layers[1])
            ]
        )
        self.norm2 = norm_layer(embed_dim[1])

        self.aux_blocks2 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + layers[0]],
                    norm_layer=norm_layer,
                )
                for i in range(layers[1])
            ]
        )
        self.aux_norm2 = norm_layer(embed_dim[1])

        if self.windows:
            print("Use local window for all blocks in stage3")
            self.blocks3 = nn.ModuleList(
                [
                    SABlock_Windows(
                        dim=embed_dim[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + layers[0] + layers[1]],
                        norm_layer=norm_layer,
                    )
                    for i in range(layers[2])
                ]
            )
            self.aux_blocks3 = nn.ModuleList(
                [
                    SABlock_Windows(
                        dim=embed_dim[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + layers[0] + layers[1]],
                        norm_layer=norm_layer,
                    )
                    for i in range(layers[2])
                ]
            )

        elif hybrid:
            print("Use hybrid window for blocks in stage3")
            block3 = []
            aux_block3 = []
            for i in range(layers[2]):
                if (i + 1) % 4 == 0:
                    block3.append(
                        SABlock(
                            dim=embed_dim[2],
                            num_heads=num_heads[2],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i + layers[0] + layers[1]],
                            norm_layer=norm_layer,
                        )
                    )
                    aux_block3.append(
                        SABlock(
                            dim=embed_dim[2],
                            num_heads=num_heads[2],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i + layers[0] + layers[1]],
                            norm_layer=norm_layer,
                        )
                    )
                else:
                    block3.append(
                        SABlock_Windows(
                            dim=embed_dim[2],
                            num_heads=num_heads[2],
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i + layers[0] + layers[1]],
                            norm_layer=norm_layer,
                        )
                    )
                    aux_block3.append(
                        SABlock_Windows(
                            dim=embed_dim[2],
                            num_heads=num_heads[2],
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[i + layers[0] + layers[1]],
                            norm_layer=norm_layer,
                        )
                    )
            self.blocks3 = nn.ModuleList(block3)
            self.aux_blocks3 = nn.ModuleList(aux_block3)
        else:
            print("Use global window for all blocks in stage3")
            self.blocks3 = nn.ModuleList(
                [
                    SABlock(
                        dim=embed_dim[2],
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + layers[0] + layers[1]],
                        norm_layer=norm_layer,
                    )
                    for i in range(layers[2])
                ]
            )
            self.aux_blocks3 = nn.ModuleList(
                [
                    SABlock(
                        dim=embed_dim[2],
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i + layers[0] + layers[1]],
                        norm_layer=norm_layer,
                    )
                    for i in range(layers[2])
                ]
            )
        self.norm3 = norm_layer(embed_dim[2])
        self.aux_norm3 = norm_layer(embed_dim[2])
        self.blocks4 = nn.ModuleList(
            [
                SABlock(
                    dim=embed_dim[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + layers[0] + layers[1] + layers[2]],
                    norm_layer=norm_layer,
                )
                for i in range(layers[3])
            ]
        )
        self.norm4 = norm_layer(embed_dim[3])
        self.aux_blocks4 = nn.ModuleList(
            [
                SABlock(
                    dim=embed_dim[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i + layers[0] + layers[1] + layers[2]],
                    norm_layer=norm_layer,
                )
                for i in range(layers[3])
            ]
        )
        self.aux_norm4 = norm_layer(embed_dim[3])

        self.FCMs = nn.ModuleList(
            [
                FCM(dim=embed_dim[0], reduction=1),
                FCM(dim=embed_dim[1], reduction=1),
                FCM(dim=embed_dim[2], reduction=1),
                FCM(dim=embed_dim[3], reduction=1),
            ]
        )

        self.FFMs = nn.ModuleList(
            [
                FFM(
                    dim=embed_dim[0],
                    reduction=1,
                    num_heads=num_heads[0],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[0],
                ),
                FFM(
                    dim=embed_dim[1],
                    reduction=1,
                    num_heads=num_heads[1],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[1],
                ),
                FFM(
                    dim=embed_dim[2],
                    reduction=1,
                    num_heads=num_heads[2],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[2],
                ),
                FFM(
                    dim=embed_dim[3],
                    reduction=1,
                    num_heads=num_heads[3],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[3],
                ),
            ]
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

    def forward_features(self, x1, x2):
        out = []

        x1 = self.patch_embed1(x1)
        x1 = self.pos_drop(x1)
        x2 = self.aux_patch_embed1(x2)
        x2 = self.aux_pos_drop(x2)

        for i, blk in enumerate(self.blocks1):
            x1 = blk(x1)

        for i, blk in enumerate(self.aux_blocks1):
            x2 = blk(x2)

        x1, x2 = self.FCMs[0](x1, x2)

        x1_1 = self.norm1(x1.permute(0, 2, 3, 1).contiguous())
        x2_1 = self.aux_norm1(x2.permute(0, 2, 3, 1).contiguous())
        fuse = self.FFMs[0](
            x1_1.permute(0, 3, 1, 2).contiguous(), x2_1.permute(0, 3, 1, 2).contiguous()
        )
        out.append(fuse)

        x1 = self.patch_embed2(x1)
        x2 = self.aux_patch_embed2(x2)
        for i, blk in enumerate(self.blocks2):
            x1 = blk(x1)

        for i, blk in enumerate(self.aux_blocks2):
            x2 = blk(x2)

        x1, x2 = self.FCMs[1](x1, x2)
        x1_1 = self.norm2(x1.permute(0, 2, 3, 1).contiguous())
        x2_1 = self.aux_norm2(x2.permute(0, 2, 3, 1).contiguous())
        fuse = self.FFMs[1](
            x1_1.permute(0, 3, 1, 2).contiguous(), x2_1.permute(0, 3, 1, 2).contiguous()
        )
        out.append(fuse)

        x1 = self.patch_embed3(x1)
        x2 = self.aux_patch_embed3(x2)
        for i, blk in enumerate(self.blocks3):
            x1 = blk(x1)

        for i, blk in enumerate(self.aux_blocks3):
            x2 = blk(x2)

        x1, x2 = self.FCMs[2](x1, x2)
        x1_1 = self.norm3(x1.permute(0, 2, 3, 1).contiguous())
        x2_1 = self.aux_norm3(x2.permute(0, 2, 3, 1).contiguous())
        fuse = self.FFMs[2](
            x1_1.permute(0, 3, 1, 2).contiguous(), x2_1.permute(0, 3, 1, 2).contiguous()
        )
        out.append(fuse)

        x1 = self.patch_embed4(x1)
        x2 = self.aux_patch_embed4(x2)
        for i, blk in enumerate(self.blocks4):
            x1 = blk(x1)

        for i, blk in enumerate(self.aux_blocks4):
            x2 = blk(x2)

        x1, x2 = self.FCMs[3](x1, x2)
        x1_1 = self.norm4(x1.permute(0, 2, 3, 1).contiguous())
        x2_1 = self.aux_norm4(x2.permute(0, 2, 3, 1).contiguous())
        fuse = self.FFMs[3](
            x1_1.permute(0, 3, 1, 2).contiguous(), x2_1.permute(0, 3, 1, 2).contiguous()
        )
        out.append(fuse)

        return tuple(out)

    def forward(self, x):
        x1, x2 = torch.split(x, 3, dim=1)
        x = self.forward_features(x1, x2)
        return x
