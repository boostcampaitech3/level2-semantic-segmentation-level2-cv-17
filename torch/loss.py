from utils import *


def get_loss(args):
    if args.criterion == 'dice':
        criterion = smp.losses.DiceLoss('multiclass')
    elif args.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.criterion == 'focal':
        criterion = smp.losses.FocalLoss('multiclass')
    elif args.criterion == 'softCE':
        criterion = smp.losses.SoftCrossEntropyLoss('multiclass')
    
    return args, criterion