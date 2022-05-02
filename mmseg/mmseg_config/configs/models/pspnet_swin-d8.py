# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)

backbone_pretrained = '/opt/ml/input/data/pretrain/swin_large_patch4_window12_384_22k.pth'

# AssertionError: EncoderDecoder: SwinTransformer: 
# 이런 게 뜨면 그 타입 부분의 parameter들을 잘 살펴보도록

model = dict(
    type='EncoderDecoder',
    pretrained=None,

    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
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
        type='PSPHead',
        in_channels=1536,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        # ignore_index=0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0 ,)),#avg_non_ignore=True)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        # ignore_index=0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,)),# avg_non_ignore=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
