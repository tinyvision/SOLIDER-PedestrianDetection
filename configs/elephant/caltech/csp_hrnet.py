# model settings
model = dict(
    type='CSP',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))
        ),
        # frozen_stages=-1,
        norm_eval=False,
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=768,
        num_outs=1),
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
            alpha=0.25,
            loss_weight=0.01),
        loss_bbox=dict(type='IoULoss', loss_weight=0.05),
        loss_offset=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.1)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.1, #0.2, #0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoCSPORIDataset'
data_root = './datasets/Caltech/test_images/'
INF = 1e8
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='./datasets/Caltech/train_m.json',
        img_prefix=data_root,

        img_scale=(640, 480),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True,
        remove_small_box=True,
        small_box_size=8,
        strides=[4],
        regress_ranges=((-1, INF),)),
    val=dict(
        type=dataset_type,
        ann_file='./datasets/Caltech/val.json',
        img_prefix=data_root,
        #img_scale=(1333, 800),
        img_scale = (640, 480),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file='./datasets/Caltech/test.json',
        img_prefix=data_root,
        img_scale=(640, 480),
        img_norm_cfg=img_norm_cfg,
        size_divisor=128,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.01/10,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
mean_teacher = True
optimizer = dict(
    type='Adam',
    lr=2e-4,
)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(mean_teacher = dict(alpha=0.999))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=250,
    warmup_ratio=1.0 / 3,
    step=[80])

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, eval_hook='CocoDistEvalMRHook')
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 120
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/netscratch/hkhan/work_dirs/csp_hrnet_caltech'
load_from = None
# load_from = '/home/ljp/code/mmdetection/work_dirs/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x/epoch_22.pth'
resume_from = None
# resume_from = '/home/ljp/code/mmdetection/work_dirs/csp4_mstrain_640_800_x101_64x4d_fpn_gn_2x/epoch_10.pth'
workflow = [('train', 1)]
