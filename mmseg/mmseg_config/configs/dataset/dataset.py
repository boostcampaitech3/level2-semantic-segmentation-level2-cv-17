# dataset settings
dataset_type = 'CustomDataset'
img_dir='/opt/ml/input/data/mmseg/img_dir/'
ann_dir= '/opt/ml/input/data/mmseg/ann_dir/'

classes = ("Backgroud","General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
palette =  [[0,0,0], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128],
           [64,0,192] ,[192,128,64], [192,192,128], [64,64,128], [128,0,192]]

img_norm_cfg = dict(
    mean = [109.9291, 117.2673, 123.4647] , std = [54.8851, 53.497 , 54.0975], to_rgb=True)
# train_all mean, std


train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(512,512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

val_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=[(512,512)],#[(1024, 1024),(512,512),(1333,800)],
            flip= False,
            flip_direction =  ["horizontal", "vertical" ],
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'train',
        img_dir=img_dir + 'train',
        classes = classes,
        palette= palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_dir=ann_dir + 'val',
        img_dir=img_dir + 'val',
        classes = classes,
        palette= palette,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        img_dir=img_dir+'test' ,
        classes = classes,
        palette= palette,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mIoU',    
                classwise=True,
                )







