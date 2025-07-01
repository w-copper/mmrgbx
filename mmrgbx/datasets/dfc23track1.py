from mmrgbx.registry import DATASETS
from .basemultisegdataset import BaseMulitInputPsSegDataset
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class DFC23Track1(BaseSegDataset):
    CLASSES = [
        "background",
        "flat_roof",
        "gable_roof",
        "gambrel_roof",
        "row_roof",
        "multiple_eave_roof",
        "hipped_roof_v1",
        "hipped_roof_v2",
        "mansard_roof",
        "pyramid_roof",
        "arched_roof",
        "dome",
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
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
    ]

    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )


@DATASETS.register_module()
class DFC23Track1WithPs(BaseMulitInputPsSegDataset):
    """
    flat_roof (31, 119, 180)
    gable_roof (174, 199, 232)
    gambrel_roof (255, 127, 14)
    row_roof (255, 187, 120)
    multiple_eave_roof (44, 160, 44)
    hipped_roof_v1 (152, 223, 138)
    hipped_roof_v2 (214, 39, 40)
    mansard_roof (255, 152, 150)
    pyramid_roof (148, 103, 189)
    arched_roof (197, 176, 213)
    dome (140, 86, 75)
    other (196, 156, 148)

    """

    CLASSES = [
        "background",
        "flat_roof",
        "gable_roof",
        "gambrel_roof",
        "row_roof",
        "multiple_eave_roof",
        "hipped_roof_v1",
        "hipped_roof_v2",
        "mansard_roof",
        "pyramid_roof",
        "arched_roof",
        "dome",
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
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
    ]

    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )

    def __init__(
        self,
        with_ps=True,
        img_suffix=".tif",
        seg_map_suffix=".tif",
        other_suffixs=dict(
            sar=".tif",
            sar_ps=".png",
        ),
        data_prefix=dict(
            img_path="rgb",
            sar="sar",
            sar_ps="sarps",
            seg_map_path="mask",
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
