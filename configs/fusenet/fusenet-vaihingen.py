_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/fusenet.py",
    "../_base_/datasets/vaihingen.py",
    "../_base_/schedule.py",
]
model = dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=[
            dict(
                type="mmseg.CrossEntropyLoss",
                use_sigmoid=False,
            ),
            dict(type="mmseg.DiceLoss", loss_weight=0.4),
        ],
    ),
)
