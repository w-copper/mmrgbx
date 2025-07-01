default_scope = "mmrgbx"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="mmseg.LocalVisBackend")]
visualizer = dict(
    type="mmseg.SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

tta_model = dict(type="mmseg.SegTTAModel")
