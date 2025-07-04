_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/ftransunet-vitb-res50.py",
    "../_base_/datasets/vaihingen_wops.py",
    "../_base_/schedule.py",
]
model = dict(
    backbone=dict(img_size=512),
    decode_head=dict(
        num_classes=6,
        loss_decode=[
            dict(type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="mmseg.DiceLoss", loss_weight=0.4),
            dict(type="mmseg.FocalLoss", loss_weight=1.0),
        ],
    ),
)
