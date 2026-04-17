# rpn_config_msfa.py
_base_ = '/home/jhu/SATLOCK/MSFA/MSFA/local_configs/SARDet/r50_dota_pretrain/fg_frcnn_dota_pretrain_sar_wavelet_r50.py'

# Keep more raw proposals from the RPN.
model = dict(
    test_cfg=dict(
        rpn=dict(
            nms_pre=12000,
            max_per_img=4000,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0,
        ),
        # RCNN settings do not matter if you only save RPN proposals,
        # but keep them loose for debugging/comparison.
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1000,
        ),
    )
)

# Used by the script below for proposal caching behavior.
rpn_cache_cfg = dict(
    max_proposals_per_image=4000,
    score_threshold=0.0,
    nms_threshold=0.9,
    save_format='pickle',
)

# Optional: match your larger test size if you want.
# Remove LoadAnnotations for plain inference on Umbra.
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]