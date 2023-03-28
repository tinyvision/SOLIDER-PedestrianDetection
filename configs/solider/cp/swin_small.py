# model settings
model = dict(
    type='CSP',
    pretrained='path/to/SOLIDER/log/lup/swin_small/checkpoin.pth',
    backbone=dict(
        type='SwinSmall',
        strides=(4, 2, 2, 1),
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        convert_weights=False,
        semantic_weight=1.0,
        ),
    neck=dict(
        type='CSPNeck',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CSPHead',
        num_classes=2,
        in_channels=768,
        stacked_convs=1,
        feat_channels=256,
        strides=[4],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25, #same to pedestron
            loss_weight=0.01),
        loss_bbox=dict(type='IoULoss', loss_weight=1), #same to pedestron
        loss_offset=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.1),
        predict_width=True,
            )
)
# training and testing settings
train_cfg = dict(
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4, #same to pedestron
            min_pos_iou=0.0, #same to pedestron
            rescale_labels=False,
            soft_labels=False,
            ignore_iof_thr=-1),
        pos_weight=-1, #same to pedestron
        debug=False,
    ),
)
test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
)
# dataset settings
dataset_type = 'CocoCSPORIDataset'
data_root = 'path/to/CityPersons/'
INF = 1e8
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        small_box_to_ignore=False,
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        remove_small_box=True,
        small_box_size=8,
        strides=[4],
        regress_ranges=((-1, INF),),
        with_width=True,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_gt_mm_nested.json',
        img_prefix=data_root + "leftImg8bit_trainvaltest/leftImg8bit/val/",
        img_scale = (2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_gt_mm_nested.json',
        img_prefix=data_root + "leftImg8bit_trainvaltest/leftImg8bit/val/",
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
mean_teacher = True
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(mean_teacher = dict(alpha=0.999))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_ratio=1.0 / 3,
    warmup_iters=500,
    step=[110,160])

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings
total_epochs = 240
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cp/swin_small/'
load_from = None
resume_from = None
workflow = [('train', 1)]
