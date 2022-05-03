import torch.nn as nn
import segmentation_models_pytorch as smp


def get_loss(loss):
    if loss == 'dice':
        return smp.losses.DiceLoss('multiclass')
    elif loss == 'CE':
        return nn.CrossEntropyLoss()
    elif loss == 'focal':
        return smp.losses.FocalLoss('multiclass')
    elif loss == 'softCE':
        return smp.losses.SoftCrossEntropyLoss('multiclass')