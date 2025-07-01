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
    type="SiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    score_weight=0.01,
    backbone=dict(
        type="ResNet",
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        style="pytorch",
    ),
    neck=dict(
        type="StackNecks",
        necks=[
            dict(
                type="SimpleFeatureFusionNeck",
                in_channels=[128, 256, 512],
                out_indices=(0, 1, 2),
            ),
            dict(
                type="mmseg.FPN",
                in_channels=[128 * 2, 256 * 2, 512 * 2],
                out_channels=256,
                num_outs=3,
            ),
        ],
    ),
    decode_head=dict(
        type="mmseg.UPerHead",
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        pool_scales=(2, 3, 6),
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
        in_channels=256,
        in_index=1,
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
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
