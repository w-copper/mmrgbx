# dataset settings
dataset_type = "C2SegWithPs"
data_root = "/scratch/wangtong/C2SegClip"
crop_size = (512, 512)
train_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgLoadAnnotations", reduce_zero_label=True, with_ps=False),
    dict(type="AnyImageToRGB"),
    dict(
        type="MultiImgRandomResize",
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5),
    dict(type="MultiImgPhotoMetricDistortion"),
    dict(type="MultiImgPackSegInputs"),
]
test_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgLoadAnnotations", reduce_zero_label=True, with_ps=False),
    dict(type="AnyImageToRGB"),
    dict(type="MultiImgPackSegInputs"),
]
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type="MultiImgLoadImageFromFile", backend_args=None),
    dict(type="AnyImageToRGB"),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(type="MultiImgResize", scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type="MultiImgRandomFlip", prob=0.0, direction="horizontal"),
                dict(type="MultiImgRandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="MultiImgLoadAnnotations")],
            [dict(type="MultiImgPackSegInputs")],
        ],
    ),
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix="_msi.tif",
        seg_map_suffix="_label.tif",
        other_suffixs=dict(
            sar="_sar.tif",
            hsi0="_hsi_0.tif",
            hsi1="_hsi_1.tif",
        ),
        with_ps=False,
        data_prefix=dict(
            seg_map_path="train",
            img_path="train",
            sar="train",
            hsi0="train",
            hsi1="train",
            sar_ps="trainps/sar",
            hsi0_ps="trainps/hsi_0",
            hsi1_ps="trainps/hsi_1",
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix="_msi.tif",
        seg_map_suffix="_label.tif",
        other_suffixs=dict(
            sar="_sar.tif",
            hsi0="_hsi_0.tif",
            hsi1="_hsi_1.tif",
        ),
        data_prefix=dict(
            seg_map_path="val",
            img_path="val",
            sar="val",
            hsi0="val",
            hsi1="val",
            sar_ps="valps/sar",
            hsi0_ps="valps/hsi_0",
            hsi1_ps="valps/hsi_1",
        ),
        with_ps=False,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="mmseg.IoUMetric", iou_metrics=["mIoU", "mFscore"])
test_evaluator = val_evaluator
