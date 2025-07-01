# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.utils import ConfigType
from mmrgbx.registry import MODELS


@MODELS.register_module()
class SimpleFeatureFusionNeck(BaseModule):
    """Feature Fusion Neck.

    Args:
        policy (str): The operation to fuse features. candidates
            are `concat`, `sum`, `diff` and `abs_diff`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(
        self,
        policy,
        in_channels=None,
        channels=None,
        out_indices=(0, 1, 2, 3),
        add_normal=False,
        norm_layer: ConfigType = None,
    ):
        super().__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.add_normal = add_normal
        if norm_layer is not None:
            self.norm_layer = MODELS.build(norm_layer)
        else:
            self.norm_layer = None

    def fusion(self, xs, policy):
        """Specify the form of feature fusion"""

        _fusion_policies = ["concat", "sum"]
        assert policy in _fusion_policies, (
            "The fusion policies {} are supported".format(_fusion_policies)
        )

        if policy == "concat":
            x = torch.cat(xs, dim=1)
        elif policy == "sum":
            x = sum(xs)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x

    def forward(self, features_per_modals):
        """Forward function."""
        outs = []
        in_len = len(features_per_modals[0])
        for i in range(in_len):
            xs = []
            for j in range(len(features_per_modals)):
                xs.append(features_per_modals[j][i])

            out = self.fusion(xs, self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
