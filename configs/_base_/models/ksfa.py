# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="MultiInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32),
)
model = dict(
    type="KsfaEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    score_weight=0.01,
    backbone=dict(
        type="KSFAEncoder",
        encoder_cfg=dict(),
        init_cfg=None,
        embed_dim=768,
        patch_size=(16, 16),
        grid_size=(16, 16),
        sampler_k=2,
        align_around=5,
        align_r=0.5,
        score_buff=10,
        ignore_index=255,
        top_k=10,
        decoder_head=8,
        dropout=0.1,
        num_layer=3,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
