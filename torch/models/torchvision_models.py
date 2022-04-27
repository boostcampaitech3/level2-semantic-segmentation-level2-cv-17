import torch.nn as nn
from torchvision import models

def get_fcn_resnet50():
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    return model