import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import CustomDataLoader, collate_fn
from utils import set_seeds

import numpy as np
import pandas as pd
from tqdm import tqdm

#!pip install albumentations==0.4.6 ?
import albumentations as A
from albumentations.pytorch import ToTensorV2


def test(model, data_loader, device):
    size = 256
    transform = A.Compose([
        A.Resize(size, size)
    ])
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def main():
    seed = 42
    set_seeds(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    category_names = ['Backgroud','General trash','Paper','Paper pack','Metal',
                    'Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

    # best model 저장된 경로
    model_path = '/opt/ml/input/code/saved/fcn_resnet50_best_model(pretrained).pt'

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    model.load_state_dict(state_dict)
    model = model.to(device)

    test_path = '/opt/ml/input/data/test.json'
    test_transform = A.Compose([
        ToTensorV2()
    ])
    test_dataset = CustomDataLoader(test_path, category_names, mode='test', transform=test_transform)
    # batch size 1로 할 경우 에러 납니다. 차원 문제로 추정
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, pin_memory=True,
                             shuffle=False, num_workers=4, collate_fn=collate_fn)

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("/opt/ml/input/code/submission/submission.csv", index=False)


if __name__ == '__main__':
    main()