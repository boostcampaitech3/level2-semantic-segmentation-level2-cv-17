from utils import *


def get_loss(args):
    if args.class_weights: class_weights = torch.tensor(args.class_weights).float().to(args.device)
    else: class_weights = None

    if args.cls_class_weights: cls_class_weights = torch.tensor(args.cls_class_weights).float().to(args.device)
    else: cls_class_weights = None
    
    if args.cls_pos_weights: cls_pos_weights = torch.tensor(args.cls_pos_weights).float().to(args.device)
    else: cls_pos_weights = None
    
    # for main criterion
    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights
            )
    elif args.criterion == 'jaccard':
        criterion = smp.losses.JaccardLoss('multiclass')
    elif args.criterion == 'dice':
        criterion = smp.losses.DiceLoss('multiclass')
    elif args.criterion == 'tversky':
        criterion = smp.losses.TverskyLoss('multiclass')
    elif args.criterion == 'focal':
        criterion = smp.losses.FocalLoss('multiclass')
    elif args.criterion == 'lovasz':
        criterion = smp.losses.LovaszLoss('multiclass')
    elif args.criterion == 'softCE':
        args.smooth_factor = 0.1
        criterion = smp.losses.SoftCrossEntropyLoss(smooth_factor=args.smooth_factor)

    # for sub criterion
    if args.sub_criterion == 'CE':
        sub_criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights
            )
    elif args.sub_criterion == 'jaccard':
        sub_criterion = smp.losses.JaccardLoss('multiclass')
    elif args.sub_criterion == 'dice':
        sub_criterion = smp.losses.DiceLoss('multiclass')
    elif args.sub_criterion == 'tversky':
        sub_criterion = smp.losses.TverskyLoss('multiclass')
    elif args.sub_criterion == 'focal':
        sub_criterion = smp.losses.FocalLoss('multiclass')
    elif args.sub_criterion == 'lovasz':
        sub_criterion = smp.losses.LovaszLoss('multiclass')
    elif args.sub_criterion == 'softCE':
        args.sub_smooth_factor = 0.1
        sub_criterion = smp.losses.SoftCrossEntropyLoss(smooth_factor=args.sub_smooth_factor)
    else:
        sub_criterion = None

    # for class criterion
    if args.cls_criterion == 'BCELogit':
        cls_criterion = torch.nn.BCEWithLogitsLoss(
            weight=cls_class_weights,
            pos_weight=cls_pos_weights
            )
    else:
        cls_criterion = None

    return args, criterion, cls_criterion, sub_criterion

