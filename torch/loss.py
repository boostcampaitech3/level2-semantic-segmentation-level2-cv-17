from utils import *


def get_loss(args):
    args.class_weights = [0.0146, 0.0948, 0.0502, 0.1144, 0.1153, 0.1065, 0.0567, 0.1077, 0.0481, 0.154, 0.1378]
    class_weights = torch.tensor(args.class_weights).float().to(args.device)
    args.pos_weights = [0.015, 0.2535, 0.0675, 1.0695, 0.924, 0.8535, 0.228, 0.432, 0.0645, 9.255, 1.8405]
    pos_weights = torch.tensor(args.pos_weights).float().to(args.device)
    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss(
            # weight=class_weights
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
    # for class criterion
    if args.cls_criterion == 'BCELogit':
        cls_criterion = torch.nn.BCEWithLogitsLoss(
            # weight=class_weights,
            # pos_weight=pos_weights
            )
    else:
        cls_criterion = None
    return args, criterion, cls_criterion