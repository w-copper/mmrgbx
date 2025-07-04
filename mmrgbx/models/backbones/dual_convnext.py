import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from .modules import FeatureFusion as FFM
from .modules import FeatureCorrection_s2c as FCM
from mmrgbx.registry import MODELS


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


@MODELS.register_module()
class DualConvNeXt(nn.Module):
    ARCH = {
        "tiny": dict(
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            nheads=[3, 6, 12, 24],
            reduction=[2, 2, 2, 2],
        ),
        "small": dict(
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
            nheads=[3, 6, 12, 24],
            reduction=[2, 2, 2, 2],
        ),
        "base": dict(
            depths=[3, 3, 27, 3],
            dims=[128, 256, 512, 1024],
            nheads=[4, 8, 16, 32],
            reduction=[2, 2, 2, 2],
        ),
        "large": dict(
            depths=[3, 3, 27, 3],
            dims=[192, 384, 768, 1536],
            nheads=[6, 12, 24, 48],
            reduction=[4, 4, 4, 4],
        ),
        "huge": dict(
            depths=[3, 3, 27, 3],
            dims=[256, 512, 1024, 2048],
            nheads=[8, 16, 32, 64],
            reduction=[4, 4, 4, 4],
        ),
    }

    def __init__(
        self,
        arch,
        in_chans: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        out_indices=[0, 1, 2, 3],
        sr_ratios=[8, 4, 2, 1],
        norm_fuse=nn.BatchNorm2d,
    ):
        super().__init__()
        self.out_indices = out_indices
        arch = self.ARCH[arch]
        depths = arch["depths"]
        dims = arch["dims"]
        nheads = arch["nheads"]
        reduction = arch["reduction"]

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        self.aux_downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        aux_stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        self.aux_downsample_layers.append(aux_stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            aux_downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            self.aux_downsample_layers.append(aux_downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple blocks
        self.aux_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            aux_stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            self.aux_stages.append(aux_stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

            aux_layer = norm_layer(dims[i_layer])
            aux_layer_name = f"aux_norm{i_layer}"
            self.add_module(aux_layer_name, aux_layer)
        self.FCMs = nn.ModuleList(
            [
                FCM(dim=dims[0], reduction=reduction[0]),
                FCM(dim=dims[1], reduction=reduction[1]),
                FCM(dim=dims[2], reduction=reduction[2]),
                FCM(dim=dims[3], reduction=reduction[3]),
            ]
        )

        self.FFMs = nn.ModuleList(
            [
                FFM(
                    dim=dims[0],
                    reduction=reduction[0],
                    num_heads=nheads[0],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[0],
                ),
                FFM(
                    dim=dims[1],
                    reduction=reduction[1],
                    num_heads=nheads[1],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[1],
                ),
                FFM(
                    dim=dims[2],
                    reduction=reduction[2],
                    num_heads=nheads[2],
                    norm_layer=norm_fuse,
                    sr_ratio=sr_ratios[2],
                ),
                FFM(
                    dim=dims[3],
                    reduction=reduction[3],
                    num_heads=nheads[3],
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
        outs = []
        for i in range(4):
            x1 = self.downsample_layers[i](x1)
            x1 = self.stages[i](x1)

            x2 = self.aux_downsample_layers[i](x2)
            x2 = self.aux_stages[i](x2)

            x1, x2 = self.FCMs[i](x1, x2)

            x1_1, x2_1 = x1, x2
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x1_1 = norm_layer(x1_1)

                aux_norm_layer = getattr(self, f"aux_norm{i}")
                x2_1 = aux_norm_layer(x2_1)

                fuse = self.FFMs[i](x1_1, x2_1)
                outs.append(fuse)

        return tuple(outs)

    def forward(self, x):
        x1, x2 = torch.split(x, 3, dim=1)[:2]
        x = self.forward_features(x1, x2)

        return x
