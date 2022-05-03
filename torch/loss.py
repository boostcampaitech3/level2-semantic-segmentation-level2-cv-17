from utils import *


def get_loss(args):
    if args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
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
    return args, criterion