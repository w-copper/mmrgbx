from mmrgbx.registry import DATASETS
from .basemultisegdataset import BaseMulitInputPsSegDataset


@DATASETS.register_module()
class PotsdamMultiClipPs(BaseMulitInputPsSegDataset):
    """Potsdam dataset."""

    CLASSES = (
        "impervious surfaces",
        "building",
        "low vegetation",
        "tree",
        "car",
        "clutter",
    )
    PALETTE = [
        [128, 128, 128],
        [128, 0, 0],
        [192, 192, 128],
        [128, 64, 128],
        [0, 0, 192],
        [128, 128, 0],
    ]

    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )

    def __init__(
        self,
        with_ps=True,
        img_suffix=".tif",
        seg_map_suffix=".png",
        other_suffixs=dict(
            depth=".tif",
            depth_ps=".tif",
        ),
        data_prefix=dict(img_path="img_dir", depth="depth_dir", depth_ps="ps_dir"),
        reduce_zero_label=True,
        **kwargs,
    ):
        super().__init__(
            with_ps=with_ps,
            other_suffixs=other_suffixs,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_prefix=data_prefix,
            **kwargs,
        )
