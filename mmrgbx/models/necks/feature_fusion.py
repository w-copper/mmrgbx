# Copyright (c) Open-CD. All rights reserved.
import torch
from mmengine.model import BaseModule
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
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        self.policy = policy
        self.out_indices = out_indices

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
        else:
            raise NotImplementedError()
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
