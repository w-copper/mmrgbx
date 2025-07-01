from mmrgbx.registry import DATASETS
from .basemultisegdataset import BaseMulitInputPsSegDataset
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class YESegOptSar(BaseSegDataset):
    CLASSES = [
        "background",
        "bare ground",
        "low vegetation",
        "trees",
        "houses",
        "water",
        "roads",
        "other",
    ]
    PALETTE = [
        [0, 0, 0],
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
    ]
    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )


@DATASETS.register_module()
class YESegOptSarWithPs(BaseMulitInputPsSegDataset):
    """
    0	background	2.26
    1	bare ground	55.84
    2	low vegetation	26.55
    3	trees	5.29
    4	houses	4.39
    5	water	1.69
    6	roads	3.88
    7	other	0.11

    """

    CLASSES = [
        "background",
        "bare ground",
        "low vegetation",
        "trees",
        "houses",
        "water",
        "roads",
        "other",
    ]
    PALETTE = [
        [0, 0, 0],
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
    ]
    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )

    def __init__(
        self,
        with_ps=True,
        img_suffix=".png",
        seg_map_suffix=".png",
        other_suffixs=dict(
            sar=".png",
            sar_ps=".png",
        ),
        data_prefix=dict(
            img_path="optical",
            sar="sar",
            sar_ps="sarps",
            seg_map_path="label",
        ),
        ignore_index=None,
        reduce_zero_label=False,
        **kwargs,
    ):
        super().__init__(
            with_ps=with_ps,
            other_suffixs=other_suffixs,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_prefix=data_prefix,
            ignore_index=ignore_index,
            **kwargs,
        )
