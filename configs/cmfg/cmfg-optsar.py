_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/cmfg.py",
    "../_base_/datasets/optsar.py",
    "../_base_/schedule.py",
]
model = dict(
    decode_head=dict(
        num_classes=8,
        loss_decode=[
            dict(
                type="mmseg.CrossEntropyLoss",
                use_sigmoid=False,
            ),
            dict(type="mmseg.DiceLoss", loss_weight=0.4),
        ],
    ),
)
