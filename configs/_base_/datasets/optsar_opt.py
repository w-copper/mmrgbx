# dataset settings
dataset_type = "YESegOptSar"
data_root = "/YESeg-OPT-SAR"
crop_size = (256, 256)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(
        type="RandomResize", scale=(256, 256), ratio_range=(0.5, 2.0), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
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
        ann_file=data_root + "/train.txt",
        data_root=data_root,
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
        ann_file=data_root + "/val.txt",
        with_ps=False,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="mmseg.IoUMetric", iou_metrics=["mIoU", "mFscore"])
test_evaluator = val_evaluator
