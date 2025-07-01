_base_ = [
    "../_base_/schedule.py",
    "../_base_/datasets/dfc23track1_rgb.py",
    "mmseg::_base_/models/upernet_r50.py",
]

model = dict(
    decode_head=dict(
        init_cfg=None,
        num_classes=13,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", loss_weight=0.4),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
