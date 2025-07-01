_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/cmfg.py",
    "../_base_/datasets/c2seg.py",
    "../_base_/schedule.py",
]
model = dict(
    decode_head=dict(
        num_classes=13,
        loss_decode=[
            dict(
                type="mmseg.CrossEntropyLoss",
                use_sigmoid=False,
            ),
            dict(type="mmseg.DiceLoss", loss_weight=0.4),
        ],
    ),
)
