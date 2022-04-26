import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset import CustomDataLoader, collate_fn
from utils import label_accuracy_score, add_hist, maybe_mkdir, set_seeds

import numpy as np
from tqdm import tqdm

#!pip install albumentations==0.4.6 ?
import albumentations as A
from albumentations.pytorch import ToTensorV2


def save_model(model, saved_dir, file_name):
    # check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, device): 
    n_class = 11
    best_loss = 9999999
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        tqdm_loader = tqdm(data_loader, desc=f'Epoch {epoch+1} Train')
        for step, (images, masks, _) in enumerate(tqdm_loader):
            images = torch.stack(images)
            masks = torch.stack(masks).long()
            
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            tqdm_loader.set_postfix({
                'Loss': round(loss.item(), 4),
                'mAcc': round(acc_cls, 4),
                'mIoU': round(mIoU, 4)
            })
        
        avg_loss = validation(epoch, model, val_loader, criterion, device)
        if avg_loss < best_loss:
            print(f"Best performance at epoch {epoch + 1} and Save model in {saved_dir}")
            best_loss = avg_loss
            save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pt')

def validation(epoch, model, data_loader, criterion, device):
    category_names_viz = ['Back','General','Paper','Paperpack','Metal',
                          'Glass','Plastic','Styrofoam','Plasticbag','Battery','Cloth']
    
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        tqdm_loader = tqdm(data_loader, desc=f'Epoch {epoch+1} Valid')
        for step, (images, masks, _) in enumerate(tqdm_loader):
            
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)
            
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
            avg_loss = total_loss / cnt
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = [{cls_name : round(IoU,2)} for IoU, cls_name in zip(IoU, category_names_viz)]

            tqdm_loader.set_postfix({
                'Loss': round(avg_loss.item(), 4),
                'mAcc': round(acc_cls, 4),
                'mIoU': round(mIoU, 4),
                'cls_IoU': IoU_by_class
            })
        
    return avg_loss

def main():
    seed = 42
    set_seeds(seed)

    category_names = ['Backgroud','General trash','Paper','Paper pack','Metal',
                      'Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

    batch_size = 16
    num_epochs = 2
    learning_rate = 0.0001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_path = '/opt/ml/input/data/train.json'
    val_path = '/opt/ml/input/data/val.json'
    train_transform = A.Compose([
        ToTensorV2()
    ])
    val_transform = A.Compose([
        ToTensorV2()
    ])
    train_dataset = CustomDataLoader(train_path, category_names, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(val_path, category_names, mode='val', transform=val_transform)

    # create own Dataset
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True,
                              shuffle=True, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, pin_memory=True,
                            shuffle=False, num_workers=8, collate_fn=collate_fn)

    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

    saved_dir = '/opt/ml/input/code/saved'
    maybe_mkdir(saved_dir)
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, device)


if __name__ == '__main__':
    main()