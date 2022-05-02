from utils import *


def build_model(args):
    decoder = getattr(smp, args.decoder)
    if args.decoder in ['Unet', 'UnetPlusPlus']:
        if args.mode == 'train':
            args['decoder_channels'] = [256, 128, 64, 32, 16]
            args['decoder_attention_type'] = None
            if not args.sweep:
                args['encoder_depth'] = 5
                args['activation'] = None
                args['aux_params'] = None
        model = decoder(
            classes=args.classes,
            encoder_name=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            activation=args.activation,
            aux_params=args.aux_params,
            
            decoder_channels=args.decoder_channels, # list length should be same with encoder_depth
            decoder_attention_type=args.decoder_attention_type,
        )
    elif args.decoder == 'MAnet':
        if args.mode == 'train':
            args['decoder_channels'] = [256, 128, 64, 32, 16]
            args['decoder_pab_channels'] = 64
            if not args.sweep:
                args['encoder_depth'] = 5
                args['activation'] = None
                args['aux_params'] = None
        model = decoder(
            classes=args.classes,
            encoder_name=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            activation=args.activation,
            aux_params=args.aux_params,

            decoder_channels=args.decoder_channels, # list length should be same with encoder_depth
            decoder_pab_channels=args.decoder_pab_channels,
        )
    elif args.decoder == 'FPN':
        if args.mode == 'train':
            args['decoder_pyramid_channels'] = 256
            args['decoder_segmentation_channels'] = 128
            args['decoder_merge_policy'] = 'add'
            args['decoder_dropout'] = 0.2
            args['upsampling'] = 4
            if not args.sweep:
                args['encoder_depth'] = 5
                args['activation'] = None
                args['aux_params'] = None
        model = decoder(
            classes=args.classes,
            encoder_name=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            activation=args.activation,
            aux_params=args.aux_params,

            decoder_pyramid_channels=args.decoder_pyramid_channels,
            decoder_segmentation_channels=args.decoder_segmentation_channels,
            decoder_merge_policy=args.decoder_merge_policy,
            decoder_dropout=args.decoder_dropout,
            upsampling=args.upsampling,
        )
    elif args.decoder == 'PSPNet':
        if args.mode == 'train':
            args['psp_out_channels'] = 512
            args['psp_dropout'] = 0.2
            args['upsampling'] = 8
            if not args.sweep:
                args['encoder_depth'] = 5
                args['activation'] = None
                args['aux_params'] = None
        model = decoder(
            classes=args.classes,
            encoder_name=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            activation=args.activation,
            aux_params=args.aux_params,

            psp_out_channels=args.psp_out_channels,
            psp_dropout=args.psp_dropout,
            upsampling=args.upsampling,
        )
    elif args.decoder == 'DeepLabV3':
        if args.mode == 'train':
            args['decoder_channels'] = 256
            args['upsampling'] = 8
            if not args.sweep:
                args['encoder_depth'] = 5
                args['activation'] = None
                args['aux_params'] = None
        model = decoder(
            classes=args.classes,
            encoder_name=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            activation=args.activation,
            aux_params=args.aux_params,

            decoder_channels=args.decoder_channels,
            upsampling=args.upsampling,
        )
    elif args.decoder == 'DeepLabV3Plus':
        if args.mode == 'train':
            args['encoder_output_stride'] = 16
            args['decoder_channels'] = 256
            args['decoder_atrous_rates'] = (12, 24, 36)
            args['upsampling'] = 8
            if not args.sweep:
                args['encoder_depth'] = 5
                args['activation'] = None
                args['aux_params'] = None
        model = decoder(
            classes=args.classes,
            encoder_name=args.encoder,
            encoder_depth=args.encoder_depth,
            encoder_weights=args.encoder_weights,
            activation=args.activation,
            aux_params=args.aux_params,
            
            encoder_output_stride=args.encoder_output_stride,
            decoder_channels=args.decoder_channels,
            decoder_atrous_rates=args.decoder_atrous_rates,
            upsampling=args.upsampling,
        )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
    
    if args.mode == 'train':
        return args, (model, preprocessing_fn)
    elif args.mode == 'test':
        return model, preprocessing_fn