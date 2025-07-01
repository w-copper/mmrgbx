from mmrgbx.registry import MODELS

from mmengine.model import BaseModule
from mmseg.utils import ConfigType
import torch.nn as nn


@MODELS.register_module()
class StackNecks(BaseModule):
    """StackNecks.

    Args:
        necks (list[ConfigType]): Configs of necks.
    """

    def __init__(self, necks: list[ConfigType]):
        super().__init__()
        self.necks = nn.ModuleList([MODELS.build(neck) for neck in necks])

    def forward(self, inputs):
        """Forward function."""
        for neck in self.necks:
            inputs = neck(inputs)
        return inputs
