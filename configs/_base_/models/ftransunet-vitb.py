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
    backbone=dict(type="FTransUnet", config="ViT-B_16", img_size=256),
    neck=None,
    decode_head=dict(
        type="mmseg.FCNHead",
        in_channels=16,
        in_index=0,
        channels=32,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=13,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss",
            use_sigmoid=False,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
