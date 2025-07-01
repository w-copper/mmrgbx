_base_ = [
    "../_base_/schedule.py",
    "../_base_/datasets/optsar_opt.py",
    "mmseg::_base_/models/segformer_mit-b0.py",
]

model = dict(
    decode_head=dict(
        init_cfg=None,
        num_classes=8,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", loss_weight=0.4),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
