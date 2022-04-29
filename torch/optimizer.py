from utils import *

def get_optimizer(optimizer, params, lr, scheduler):

    if scheduler == 'cosign':
        lr = 0

    if optimizer == 'Adam':
        return torch.optim.Adam(params=params, lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(params=params, lr=lr)
    elif optimizer == 'SGD':
        return torch.optim.SGD(params=params, lr=lr)
