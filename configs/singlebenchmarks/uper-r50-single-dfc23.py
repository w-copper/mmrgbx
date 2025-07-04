_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/schedule.py",
    "../_base_/datasets/dfc23track1_rgb.py",
    "mmseg::_base_/models/upernet_r50.py",
]

model = dict(
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
    ),
    decode_head=dict(
        init_cfg=None,
        num_classes=13,
        loss_decode=[
            dict(type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="mmseg.DiceLoss", loss_weight=0.4),
            dict(type="mmseg.FocalLoss", loss_weight=1.0),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
