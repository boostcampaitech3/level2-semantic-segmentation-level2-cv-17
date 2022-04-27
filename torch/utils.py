import os
import random
from statistics import mean

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from munch import Munch
import glob
from pathlib import Path
import re


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        print(f'dirs: {dirs}')
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]
        print(f'matches: {matches}')
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        workdir = f"{path}{n}"
        return workdir

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def concat_config(arg, config):
    config = Munch(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['seed'] = arg.seed
    config['name'] = arg.name
    config['work_dir'] = arg.work_dir
    if not os.path.exists(os.path.join(os.getcwd(), arg.work_dir)):
        dir_name = os.path.join(os.getcwd(), arg.work_dir, f'exp_0')
        os.makedirs(dir_name)
    else:
        i = 1
        dir_name = os.path.join(os.getcwd(), arg.work_dir, f'exp_{i}')
        while os.path.exists(dir_name):
            i += 1
            dir_name = os.path.join(os.getcwd(), arg.work_dir, f'exp_{i}')
        os.makedirs(dir_name)
    config['work_dir_exp'] = dir_name
    with open(os.path.join(dir_name, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)
    config['save_dir'] = dir_name
    config['device'] = device
    config['data_dir'] = arg.data_dir
    config['viz_log'] = arg.viz_log
    config['metric'] = arg.metric
    config['loss'] = arg.loss
    config['save_interval'] = arg.save_interval

    return config


def load_config(args):
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)
    config = concat_config(args, config)
    return config


def get_metrics(output, mask):
    with torch.no_grad():
        output_met = torch.argmax(F.softmax(output, dim=1), dim=1) - 1
        mask_met = mask - 1
        tp, fp, fn, tn = smp.metrics.get_stats(output_met, mask_met, mode='multiclass', num_classes=10, ignore_index=-1)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    return f1_score, recall, precision


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, classwise=False, smooth=1e-10, n_classes=11):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas
            intersect = torch.logical_and(true_class, true_label).sum().item()
            union = torch.logical_or(true_class, true_label).sum().item()
            
            iou = (intersect+smooth) / (union+smooth)
            iou_per_class.append(iou)
        
        if classwise == False:
            return mean(iou_per_class)
        else:
            return iou_per_class

