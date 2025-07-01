import torch
from torch import nn, Tensor
import math

from timm.models.layers import trunc_normal_
from .modules import FeatureFusion as FFM
from .modules import FeatureCorrection_s2c as FCM
from mmrgbx.registry import MODELS


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, c1, c2, s, expand_ratio):
        super().__init__()
        ch = int(round(c1 * expand_ratio))
        self.use_res_connect = s == 1 and c1 == c2

        layers = []

        if expand_ratio != 1:
            layers.append(ConvModule(c1, ch, 1))

        layers.extend(
            [
                ConvModule(ch, ch, 3, s, 1, g=ch),
                nn.Conv2d(ch, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@MODELS.register_module()
class DualMobileNetV2(nn.Module):
    def __init__(self, norm_fuse=nn.BatchNorm2d):
        super().__init__()
        self.channels = [24, 32, 96, 320]
        input_channel = 32

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])
        self.aux_features = nn.ModuleList([ConvModule(3, input_channel, 3, 2, 1)])

        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                self.aux_features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        self.FCMs = nn.ModuleList(
            [
                FCM(dim=self.channels[0], reduction=1),
                FCM(dim=self.channels[1], reduction=1),
                FCM(dim=self.channels[2], reduction=1),
                FCM(dim=self.channels[3], reduction=1),
            ]
        )

        self.FFMs = nn.ModuleList(
            [
                FFM(
                    dim=self.channels[0],
                    reduction=1,
                    num_heads=1,
                    norm_layer=norm_fuse,
                    sr_ratio=4,
                ),
                FFM(
                    dim=self.channels[1],
                    reduction=1,
                    num_heads=2,
                    norm_layer=norm_fuse,
                    sr_ratio=3,
                ),
                FFM(
                    dim=self.channels[2],
                    reduction=1,
                    num_heads=3,
                    norm_layer=norm_fuse,
                    sr_ratio=2,
                ),
                FFM(
                    dim=self.channels[3],
                    reduction=1,
                    num_heads=4,
                    norm_layer=norm_fuse,
                    sr_ratio=1,
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

    def forward(self, x):
        x1, x2 = torch.split(x, 3, 1)
        outs = []

        for block, aux_block, fcm, ffm in zip(
            [
                self.features[:4],
                self.features[4:7],
                self.features[7:14],
                self.features[14:],
            ],
            [
                self.aux_features[:4],
                self.aux_features[4:7],
                self.aux_features[7:14],
                self.aux_features[14:],
            ],
            [self.FCMs[0], self.FCMs[1], self.FCMs[2], self.FCMs[3]],
            [self.FFMs[0], self.FFMs[1], self.FFMs[2], self.FFMs[3]],
        ):
            for blk in block:
                x1 = blk(x1)
            for blk in aux_block:
                x2 = blk(x2)
            x1, x2 = fcm(x1, x2)
            fuse = ffm(x1, x2)
            outs.append(fuse)

        return tuple(outs)
