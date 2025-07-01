import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmrgbx.registry import MODELS


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid(),
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


"""
Feature Separation Part
"""


class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out


"""
SA-Gate
"""


class SAGate(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(SAGate, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.fsp_rgb = FSP(in_planes, out_planes, reduction)
        self.fsp_hha = FSP(in_planes, out_planes, reduction)

        self.gate_rgb = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_hha = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        rgb, hha = x

        rec_rgb = self.fsp_rgb(hha, rgb)
        rec_hha = self.fsp_hha(rgb, hha)

        cat_fea = torch.cat([rec_rgb, rec_hha], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_hha(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = (
            attention_vector[:, 0:1, :, :],
            attention_vector[:, 1:2, :, :],
        )
        merge_feature = rgb * attention_vector_l + hha * attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        hha_out = (hha + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        return [rgb_out, hha_out], merge_feature


class DualBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_layer=None,
        bn_eps=1e-5,
        bn_momentum=0.1,
        downsample=None,
        inplace=True,
    ):
        super(DualBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.hha_conv1 = conv3x3(inplanes, planes, stride)
        self.hha_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.hha_relu_inplace = nn.ReLU(inplace=True)
        self.hha_conv2 = conv3x3(planes, planes)
        self.hha_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)

        self.downsample = downsample
        self.hha_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]

        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.hha_conv1(x2)
        out2 = self.hha_bn1(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv2(out2)
        out2 = self.hha_bn2(out2)

        if self.hha_downsample is not None:
            residual2 = self.hha_downsample(x2)

        out1 += residual1
        out2 += residual2

        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


class DualBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_layer=None,
        bn_eps=1e-5,
        bn_momentum=0.1,
        downsample=None,
        inplace=True,
    ):
        super(DualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.hha_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.hha_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.hha_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.hha_bn3 = norm_layer(
            planes * self.expansion, eps=bn_eps, momentum=bn_momentum
        )
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.hha_relu_inplace = nn.ReLU(inplace=True)
        self.hha_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.hha_conv1(x2)
        out2 = self.hha_bn1(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv2(out2)
        out2 = self.hha_bn2(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv3(out2)
        out2 = self.hha_bn3(out2)

        if self.hha_downsample is not None:
            residual2 = self.hha_downsample(x2)

        out1 += residual1
        out2 += residual2
        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


@MODELS.register_module()
class SAGateDualResnet(BaseModule):
    ARCH = {
        "18": dict(
            layers=[2, 2, 2, 2],
            block="DualBasicBlock",
        ),
        "34": dict(
            layers=[3, 4, 6, 3],
            block="DualBasicBlock",
        ),
        "50": dict(
            layers=[3, 4, 6, 3],
            block="DualBottleneck",
        ),
        "101": dict(
            layers=[3, 4, 23, 3],
            block="DualBottleneck",
        ),
        "152": dict(
            layers=[3, 8, 36, 3],
            block="DualBottleneck",
        ),
    }

    def __init__(
        self,
        arch,
        norm_layer=nn.BatchNorm2d,
        bn_eps=1e-5,
        bn_momentum=0.1,
        deep_stem=False,
        stem_width=32,
        inplace=True,
        init_cfg=None,
    ):
        super(SAGateDualResnet, self).__init__(init_cfg)
        self.inplanes = stem_width * 2 if deep_stem else 64
        assert arch in self.ARCH.keys(), f"arch {arch} is not supported"
        self.arch = self.ARCH[arch]
        block = self.arch["block"]
        layers = self.arch["layers"]
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    3, stem_width, kernel_size=3, stride=2, padding=1, bias=False
                ),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(
                    stem_width,
                    stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(
                    stem_width,
                    stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
            self.hha_conv1 = nn.Sequential(
                nn.Conv2d(
                    3, stem_width, kernel_size=3, stride=2, padding=1, bias=False
                ),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(
                    stem_width,
                    stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(
                    stem_width,
                    stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.hha_conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        block_map = {"DualBasicBlock": DualBasicBlock, "DualBottleneck": DualBottleneck}
        block = block_map[block]

        self.bn1 = norm_layer(
            stem_width * 2 if deep_stem else 64, eps=bn_eps, momentum=bn_momentum
        )
        self.hha_bn1 = norm_layer(
            stem_width * 2 if deep_stem else 64, eps=bn_eps, momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=inplace)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.hha_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            norm_layer,
            64,
            layers[0],
            inplace,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.layer2 = self._make_layer(
            block,
            norm_layer,
            128,
            layers[1],
            inplace,
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.layer3 = self._make_layer(
            block,
            norm_layer,
            256,
            layers[2],
            inplace,
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.layer4 = self._make_layer(
            block,
            norm_layer,
            512,
            layers[3],
            inplace,
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

        self.sagates = nn.ModuleList(
            [
                SAGate(in_planes=128, out_planes=256, bn_momentum=bn_momentum),
                SAGate(in_planes=256, out_planes=512, bn_momentum=bn_momentum),
                SAGate(in_planes=512, out_planes=1024, bn_momentum=bn_momentum),
                SAGate(in_planes=1024, out_planes=2048, bn_momentum=bn_momentum),
            ]
        )

    def _make_layer(
        self,
        block,
        norm_layer,
        planes,
        blocks,
        inplace=True,
        stride=1,
        bn_eps=1e-5,
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * block.expansion, eps=bn_eps, momentum=bn_momentum),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                norm_layer,
                bn_eps,
                bn_momentum,
                downsample,
                inplace,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                    inplace=inplace,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x1, x2 = torch.split(x, 3, dim=1)[:2]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.hha_conv1(x2)
        x2 = self.hha_bn1(x2)
        x2 = self.hha_relu(x2)
        x2 = self.hha_maxpool(x2)

        x = [x1, x2]
        blocks = []
        merges = []
        x = self.layer1(x)
        x, merge = self.sagates[0](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer2(x)
        x, merge = self.sagates[1](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer3(x)
        x, merge = self.sagates[2](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer4(x)
        x, merge = self.sagates[3](x)
        blocks.append(x)
        merges.append(merge)

        return merges
