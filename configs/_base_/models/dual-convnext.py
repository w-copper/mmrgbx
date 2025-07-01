norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="MultiInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32),
)
model = dict(
    type="MiscEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="DualConvNeXt",
        arch="small",
        out_indices=(0, 1, 2, 3),
    ),
    neck=None,
    decode_head=dict(
        type="mmseg.UPerHead",
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=13,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    auxiliary_head=dict(
        type="mmseg.FCNHead",
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=13,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
