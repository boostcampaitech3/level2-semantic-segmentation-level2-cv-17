import segmentation_models_pytorch as smp


def build_model(args):
    decoder = getattr(smp, args.decoder)
    if args.decoder in ['Unet', 'UnetPlusPlus']:
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16], # list length should be same with encoder_depth
            decoder_attention_type=None,
            classes=args.classes,
            activation=None,
        )
    elif args.decoder == 'MAnet':
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16], # list length should be same with encoder_depth
            decoder_pab_channels=64,
            classes=args.classes,
            activation=None,
        )
    elif args.decoder == 'FPN':
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_depth=5,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_dropout=0.2,
            classes=args.classes,
            activation=None,
            upsampling=4,
        )
    elif args.decoder == 'PSPNet':
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_depth=3,
            psp_out_channels=512,
            psp_dropout=0.2,
            classes=args.classes,
            activation=None,
            upsampling=8,
        )
    elif args.decoder == 'PAN':
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_output_stride=16,
            decoder_channels=32,
            classes=args.classes,
            activation=None,
            upsampling=4,
        )
    elif args.decoder == 'DeepLabV3':
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_depth=5,
            decoder_channels=256,
            classes=args.classes,
            activation=None,
            upsampling=8,
        )
    elif args.decoder == 'DeepLabV3Plus':
        model = decoder(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            encoder_depth=5,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            classes=args.classes,
            activation=None,
            upsampling=4,
        )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    
    return model, preprocessing_fn
