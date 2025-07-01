# coding=utf-8
import copy
import math

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from mmengine.config import ConfigDict
from collections import OrderedDict
import torch.nn.functional as F
from mmrgbx.registry import MODELS


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in, activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in, activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups
    )


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(
            cmid, cmid, stride, bias=False
        )  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            4, width, kernel_size=7, stride=2, bias=False, padding=3
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device
                )
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class FuseResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.activation = nn.ReLU(inplace=True)

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, width, kernel_size=7, stride=2, bias=False, padding=3
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.rootd = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, width, kernel_size=7, stride=2, bias=False, padding=3
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.se_layer0 = SqueezeAndExciteFusionAdd(64, activation=self.activation)
        self.se_layer1 = SqueezeAndExciteFusionAdd(256, activation=self.activation)
        self.se_layer2 = SqueezeAndExciteFusionAdd(512, activation=self.activation)
        self.se_layer3 = SqueezeAndExciteFusionAdd(1024, activation=self.activation)

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

        self.bodyd = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x, y):
        SE = True
        # SE = False
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)  # 64*128
        y = self.rootd(y)  # 64*128
        if SE:
            x = self.se_layer0(x, y)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        y = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(y)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)  # 256*63, 512*32
            y = self.bodyd[i](y)  # 256*63, 512*32
            if SE:
                if i == 0:
                    x = self.se_layer1(x, y)
                if i == 1:
                    x = self.se_layer2(x, y)

            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device
                )
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)  # 1024*16
        y = self.bodyd[-1](y)  # 1024*16
        if SE:
            x = self.se_layer3(x, y)
        return x, y, features[::-1]


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}


class Attention(nn.Module):
    def __init__(self, config, vis, mode=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.mode = mode
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.queryd = Linear(config.hidden_size, self.all_head_size)
        self.keyd = Linear(config.hidden_size, self.all_head_size)
        self.valued = Linear(config.hidden_size, self.all_head_size)
        self.outd = Linear(config.hidden_size, config.hidden_size)

        if self.mode == "mba":
            self.w11 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)

            # self.gate_sx = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)
            # self.gate_cx = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)
            # self.gate_sy = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)
            # self.gate_cy = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_statesx, hidden_statesy):
        mixed_query_layer = self.query(hidden_statesx)
        mixed_key_layer = self.key(hidden_statesx)
        mixed_value_layer = self.value(hidden_statesx)

        mixed_queryd_layer = self.queryd(hidden_statesy)
        mixed_keyd_layer = self.keyd(hidden_statesy)
        mixed_valued_layer = self.valued(hidden_statesy)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        queryd_layer = self.transpose_for_scores(mixed_queryd_layer)
        keyd_layer = self.transpose_for_scores(mixed_keyd_layer)
        valued_layer = self.transpose_for_scores(mixed_valued_layer)

        ## Self Attention x: Qx, Kx, Vx
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sx = self.out(context_layer)
        attention_sx = self.proj_dropout(attention_sx)

        ## Self Attention y: Qy, Ky, Vy
        attention_scores = torch.matmul(queryd_layer, keyd_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, valued_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sy = self.outd(context_layer)
        attention_sy = self.proj_dropout(attention_sy)

        # return attention_sx, attention_sy, weights
        if self.mode == "mba":
            # ## Cross Attention x: Qx, Ky, Vy
            attention_scores = torch.matmul(query_layer, keyd_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, valued_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cx = self.out(context_layer)
            attention_cx = self.proj_dropout(attention_cx)

            ## Cross Attention y: Qy, Kx, Vx
            attention_scores = torch.matmul(queryd_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cy = self.outd(context_layer)
            attention_cy = self.proj_dropout(attention_cy)

            # return attention_cx, attention_cy, weights

            # ## ADD
            # attention_x = torch.div(torch.add(attention_sx, attention_cx), 2)
            # attention_y = torch.div(torch.add(attention_sy, attention_cy), 2)
            # Adaptative MBA
            attention_sx = self.w11 * attention_sx + self.w12 * attention_cx
            attention_sy = self.w21 * attention_sy + self.w22 * attention_cy
            ## Gated MBA
            # attention_x = self.w11 * attention_sx + (1 - self.w11) * attention_cx
            # attention_y = self.w21 * attention_sy + (1 - self.w21) * attention_cy
            ## SA-GATE MBA
            # attention_sx =  attention_sx.transpose(-1, -2)
            # attention_cx =  attention_cx.transpose(-1, -2)
            # attention_sy =  attention_sy.transpose(-1, -2)
            # attention_cy =  attention_cy.transpose(-1, -2)
            # attention_vector_sx = self.gate_sx(attention_sx)
            # attention_vector_cx = self.gate_cx(attention_cx)
            # attention_vector_sy = self.gate_sy(attention_sy)
            # attention_vector_cy = self.gate_cy(attention_cy)
            # attention_vector_x = torch.cat([attention_vector_sx, attention_vector_cx], dim=1)
            # attention_vector_x = self.softmax(attention_vector_x)
            # attention_vector_y = torch.cat([attention_vector_sy, attention_vector_cy], dim=1)
            # attention_vector_y = self.softmax(attention_vector_y)

            # attention_vector_sx, attention_vector_cx = attention_vector_x[:, 0:1, :], attention_vector_x[:, 1:2, :]
            # attention_x = (attention_sx*attention_vector_sx + attention_cx*attention_vector_cx).transpose(-1, -2)
            # attention_vector_sy, attention_vector_cy = attention_vector_y[:, 0:1, :], attention_vector_y[:, 1:2, :]
            # attention_y = (attention_sy*attention_vector_sy + attention_cy*attention_vector_cy).transpose(-1, -2)

        return attention_sx, attention_sy, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (
                img_size[1] // patch_size_real[1]
            )
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = FuseResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_embeddingsd = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, y):
        # y = y.unsqueeze(1)
        if self.hybrid:
            x, y, features = self.hybrid_model(x, y)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        y = self.patch_embeddingsd(y)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        y = y.flatten(2)
        y = y.transpose(-1, -2)

        embeddingsx = x + self.position_embeddings
        embeddingsx = self.dropout(embeddingsx)
        embeddingsy = y + self.position_embeddings
        embeddingsy = self.dropout(embeddingsy)
        return embeddingsx, embeddingsy, features


class Block(nn.Module):
    def __init__(self, config, vis, mode=None):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.ffnd = Mlp(config)
        self.attn = Attention(config, vis, mode=mode)

    def forward(self, x, y):
        hx = x
        hy = y
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y, weights = self.attn(x, y)
        x = x + hx
        y = y + hy

        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        return x, y, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder_normd = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            ## 12+0
            # if i >= 0 :
            ## 3+6+3
            if i < 3 or i > 8:
                # ## 1+1+1+1...
                # if i % 2 == 0:
                layer = Block(config, vis, mode="sa")
            else:
                layer = Block(config, vis, mode="mba")
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_statesx, hidden_statesy):
        attn_weights = []
        for layer_block in self.layer:
            hidden_statesx, hidden_statesy, weights = layer_block(
                hidden_statesx, hidden_statesy
            )
            if self.vis:
                attn_weights.append(weights)
        encodedx = self.encoder_norm(hidden_statesx)
        encodedy = self.encoder_normd(hidden_statesy)
        return encodedx, encodedy, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, dsm_ids):
        embeddingsx, embeddingsy, features = self.embeddings(input_ids, dsm_ids)
        encodedx, encodedy, attn_weights = self.encoder(
            embeddingsx, embeddingsy
        )  # (B, n_patch, hidden)
        return encodedx, encodedy, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(
                4 - self.config.n_skip
            ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = (
            hidden_states.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ConfigDict()
    config.patches = ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = "seg"
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.patch_size = 16
    config.n_skip = 0

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ConfigDict()
    config.patches = ConfigDict({"size": (16, 16)})
    config.hidden_size = 1
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = ConfigDict()
    config.patches = ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = "seg"
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.patch_size = 16
    config.n_skip = 0

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = "softmax"
    config.patches.grid = (16, 16)
    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = "seg"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = "softmax"

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ConfigDict()
    config.patches = ConfigDict({"size": (16, 16)})
    config.hidden_size = 1024
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = "seg"
    config.resnet_pretrained_path = None
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized"""
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = "seg"

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ConfigDict()
    config.patches = ConfigDict({"size": (14, 14)})
    config.hidden_size = 1280
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None

    return config


@MODELS.register_module()
class FTransUnet(nn.Module):
    CONFIGS = {
        "ViT-B_16": get_b16_config(),
        "ViT-B_32": get_b32_config(),
        "ViT-L_16": get_l16_config(),
        "ViT-L_32": get_l32_config(),
        "ViT-H_14": get_h14_config(),
        "R50-ViT-B_16": get_r50_b16_config(),
        "R50-ViT-L_16": get_r50_l16_config(),
    }

    def __init__(self, config, img_size=224, zero_head=False, vis=False):
        super(FTransUnet, self).__init__()
        config = self.CONFIGS[config]
        self.zero_head = zero_head
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.config = config

    def forward(self, x):
        x, y = torch.split(x, 3, dim=1)
        x, y, attn_weights, features = self.transformer(x, y)  # (B, n_patch, hidden)
        x = x + y
        x = self.decoder(x, features)
        return [x]
