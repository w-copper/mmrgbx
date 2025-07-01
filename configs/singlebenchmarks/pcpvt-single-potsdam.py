_base_ = [
    "../_base_/schedule.py",
    "../_base_/datasets/potsdam_rgb.py",
    "mmseg::_base_/models/twins_pcpvt-s_upernet.py",
]

model = dict(
    decode_head=dict(
        init_cfg=None,
        num_classes=6,
        pretrain_img_size=512,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", loss_weight=0.4),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
