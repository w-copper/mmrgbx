optimizer = dict(type="AdamW", lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type="SkipWaveOptimWrapper",
    optimizer=optimizer,
    acc_count=30,
    alpha=0.1,
    clip_grad=dict(type="norm", max_norm=0.5, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            ".ln": dict(decay_mult=0.0),
            ".bias": dict(decay_mult=0.0),
            ".pos_embed": dict(decay_mult=0.0),
            "backbone.fusion_module.trasnformer.": dict(lr_mult=0.05),
        },
    ),
)

# learning rate scheduler
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=1e-7,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=1
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)
