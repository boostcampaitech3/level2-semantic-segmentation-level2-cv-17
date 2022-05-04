import os
import random
import yaml
import numpy as np

import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp

from munch import Munch


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_exp_dir(work_dir):
    work_dir = work_dir.split('./')[-1]
    if not os.path.exists(os.path.join(os.getcwd(), work_dir)):
        exp_dir = os.path.join(os.getcwd(), work_dir, 'exp0')
    else:
        idx = 1
        exp_dir = os.path.join(os.getcwd(), work_dir, f'exp{idx}')
        while os.path.exists(exp_dir):
            idx += 1
            exp_dir = os.path.join(os.getcwd(), work_dir, f'exp{idx}')
    
    os.makedirs(exp_dir)
    return exp_dir

# config에 args 정보를 덮어 씌우고 config를 return
def concat_config(args, config):
    config = Munch(config)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['mode'] = args.mode
    config['seed'] = args.seed
    
    config['data_dir'] = args.data_dir
    config['work_dir'] = args.work_dir
    config['work_dir_exp'] = args.work_dir_exp
    config['src_config'] = args.src_config
    config['src_config_dir'] = args.src_config_dir
    config['dst_config'] = args.dst_config
    config['dst_config_dir'] = args.dst_config_dir

    if config['mode'] == 'train':
        config['add_train'] = args.add_train
        config['save_interval'] = args.save_interval
        config['train_image_log'] = args.train_image_log
        config['valid_image_log'] = args.valid_image_log
        config['wandb_remark'] = args.wandb_remark
        config['sweep'] = args.sweep
        config['sweep_name'] = args.sweep_name
    
    else:
        config['ckpt_name'] = args.ckpt_name
        config['ckpt_dir'] = args.ckpt_dir
        config['save_remark'] = args.save_remark
    
    return config


def save_config(args, save_dir):
    with open(save_dir, 'w') as f:
        yaml.safe_dump(args, f)


def load_config(args):
    with open(args.src_config_dir, 'r') as f:
        config = yaml.safe_load(f)
    return config


def maybe_apply_remark(dir, remark, extension):
    if remark != '':
        new_name = dir[:-len(extension)] + '_' + remark
    else:
        new_name = dir[:-len(extension)]
    new_name += extension
    return new_name


def get_metrics(output, mask):
    with torch.no_grad():
        output_met = torch.argmax(F.softmax(output, dim=1), dim=1) - 1
        mask_met = mask - 1
        tp, fp, fn, tn = smp.metrics.get_stats(output_met, mask_met, mode='multiclass', num_classes=10, ignore_index=-1)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
    return f1_score, recall, precision


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """
    with torch.no_grad():
        label_preds = torch.argmax(label_preds, dim=1).detach().cpu().numpy()
        label_trues = label_trues.detach().cpu().numpy()
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist
