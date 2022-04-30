# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import pickle
import cv2

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes

warnings.filterwarnings(action='ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config',help='test config file path. (EX) pspnet.py')
    parser.add_argument('checkpoint',help='checkpoint file. (EX) exp9/epoch_45.pth')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_root = '/opt/ml/input/level2-semantic-segmentation-level2-cv-17/mmseg/mmseg_config/configs/_base_'
    checkpoint_root = '/opt/ml/input/level2-semantic-segmentation-level2-cv-17/mmseg/mmseg_config/work_dirs'


    cfg = mmcv.Config.fromfile(os.path.join(config_root,args.config))

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # if you want to use TTA 
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True


    cfg.gpu_ids = [0]

   
    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    # The default loader config
    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        shuffle=False)

    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1, # Must set 1 for some augmentations which doesn't work
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # checkpoint load
    load_checkpoint(model, os.path.join(checkpoint_root,args.checkpoint), map_location='cpu')

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    results = single_gpu_test(
            model,
            data_loader,
            # If you want to show output with saved image
            # args.show,
            # args.show_dir,
            # False,
            # args.opacity,
            )


    # submissions 
    size = 256
    results = np.array(results)

    prediction_strings = []
    file_names = []
    coco = COCO('/opt/ml/input/data/test.json')
    for i, out in enumerate(results):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        out = cv2.resize(out, (256,256), interpolation =cv2.INTER_NEAREST)
        out = out.reshape([size*size]).astype(int)

        prediction_string = ' '.join(str(e) for e in out.tolist())

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['PredictionString'] = prediction_strings
 

    sub_name = args.checkpoint[:-4] + '.csv'
    submission_path = os.path.join(checkpoint_root,sub_name)
    submission.to_csv(submission_path, index=False)


if __name__ == '__main__':
    main()
