from mmrgbx.registry import DATASETS
from .basemultisegdataset import BaseMulitInputPsSegDataset
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class C2SegDataset(BaseSegDataset):
    CLASSES = [
        "Surface water",
        "Street network",
        "Urban fabric",
        "Industrial, commercial, and transport",
        "Mine, dump, and construction sites",
        "Artificial vegetated areas",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Forests",
        "Shrub",
        "Open spaces with no vegetation",
        "Inland wetlands",
    ]
    PALETTE = [
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
        [227, 119, 194],
    ]

    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )


@DATASETS.register_module()
class C2SegWithPs(BaseMulitInputPsSegDataset):
    """
    Surface water: [31, 119, 180]
    Street network: [174, 199, 232]
    Urban fabric: [255, 127, 14]
    Industrial, commercial, and transport: [255, 187, 120]
    Mine, dump, and construction sites: [44, 160, 44]
    Artificial vegetated areas: [152, 223, 138]
    Arable land: [214, 39, 40]
    Permanent crops: [255, 152, 150]
    Pastures: [148, 103, 189]
    Forests: [197, 176, 213]
    Shrub: [140, 86, 75]
    Open spaces with no vegetation: [196, 156, 148]
    Inland wetlands: [227, 119, 194]

    """

    CLASSES = [
        "Surface water",
        "Street network",
        "Urban fabric",
        "Industrial, commercial, and transport",
        "Mine, dump, and construction sites",
        "Artificial vegetated areas",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Forests",
        "Shrub",
        "Open spaces with no vegetation",
        "Inland wetlands",
    ]
    PALETTE = [
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
        [227, 119, 194],
    ]

    METAINFO = dict(
        classes=CLASSES,
        palette=PALETTE,
    )

    def __init__(
        self,
        with_ps=True,
        img_suffix="_msi.tif",
        seg_map_suffix="_label.tif",
        other_suffixs=dict(
            sar="_sar.tif",
            hsi0="_hsi_0.tif",
            hsi1="_hsi_1.tif",
            sar_ps="_sar.png",
            hsi0_ps="_hsi_0.png",
            hsi1_ps="_hsi_1.png",
        ),
        data_prefix=dict(
            img_path="",
            sar="",
            sar_ps="",
            seg_map_path="",
        ),
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
