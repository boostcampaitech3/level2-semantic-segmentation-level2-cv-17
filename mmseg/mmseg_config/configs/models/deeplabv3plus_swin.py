# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)

backbone_pretrained = '/opt/ml/input/data/pretrain/swin_large_patch4_window12_384_22k.pth'


model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=1536,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.,
        patch_size=4,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_ratio=4,
        norm_cfg=backbone_norm_cfg,
        act_cfg=dict(type='GELU'),
        init_cfg = dict(type="Pretrained",checkpoint=backbone_pretrained)
        ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=1536,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=192,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=192,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
