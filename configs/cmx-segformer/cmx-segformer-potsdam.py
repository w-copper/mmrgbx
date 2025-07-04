_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/cmx-segformer.py",
    "../_base_/datasets/potsdam_wops.py",
    "../_base_/schedule.py",
]
model = dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=[
            dict(type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="mmseg.DiceLoss", loss_weight=0.4),
            dict(type="mmseg.FocalLoss", loss_weight=1.0),
        ],
    ),
)
