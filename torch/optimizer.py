from utils import *

def get_optimizer(args, params):

    if args.scheduler == 'cosign':
        args.scheduler_lr_max = args.lr
        args.lr = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=params, lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=params, lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=params, lr=args.lr)

    return args, optimizer