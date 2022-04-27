import os
import cv2
import numpy as np
import albumentations as albu
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2

dataset_path = '/opt/ml/input/data'


def get_classname(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


def get_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]

    return albu.Compose(train_transform)


# def get_val_transform():
#     val_transform = [
#
#     ]
#
#     return albu.Compose(val_transform)


# 안씀
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ]
    return albu.Compose(_transform)


# 안 씀
def one_hot_label(mask, classes: int):
    n_mask = torch.from_numpy(mask).long()
    shape = n_mask.shape
    one_hot = torch.zeros((classes,) + shape[0:])
    n_mask = one_hot.scatter_(1, n_mask.unsqueeze(0), 1.0)
    return n_mask


class CustomDataLoader(Dataset):
    """COCO format"""
    CLASSES = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam',
               'Plastic bag', 'Battery', 'Clothing']

    def __init__(self, data_dir, mode='train', transform=None, preprocessing=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.preprocessing = get_preprocessing(preprocessing)
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)


        if self.mode in ('train', 'val'):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
            for i in range(len(anns)):
                class_name = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.CLASSES.index(class_name)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            images = images.astype(np.float32)
            if self.preprocessing:
                transformed = self.preprocessing(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images.float(), masks.long(),

        elif self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            if self.preprocessing:
                sample = self.preprocessing(image=images)
                images = sample['image']
            return images.float()

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def load_dataset(args, preprocessing_fn):
    data_dir = args.data_dir
    train_transform = get_transform()
    train_dataset = CustomDataLoader(data_dir=os.path.join(data_dir, 'train.json'), mode='train',
                                     transform=train_transform, preprocessing=preprocessing_fn)
    val_dataset = CustomDataLoader(data_dir=os.path.join(data_dir, 'val.json'), mode='val',
                                   preprocessing=preprocessing_fn)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker)
    return train_dataloader, val_dataloader
