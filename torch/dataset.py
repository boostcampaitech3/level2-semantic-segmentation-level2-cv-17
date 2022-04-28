import os
import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from transform import get_train_transform, get_valid_transform

dataset_path = '/opt/ml/input/data'


def get_classname(class_id, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""
    CLASSES = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
               'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    def __init__(self, data_dir, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0


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
            anns = sorted(anns, key=lambda idx: len(idx['segmentation'][0]), reverse=False)

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
            return images, masks

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def collate_fn(batch):
    return tuple(zip(*batch))


def load_dataset(args, preprocessing_fn):
    data_dir = args.data_dir

    # transform.py에 있는 custom augmentation 함수 사용
    train_transform = get_train_transform(preprocessing_fn)
    val_transform = get_valid_transform(preprocessing_fn)

    train_dataset = CustomDataLoader(data_dir=os.path.join(data_dir, 'train.json'), mode='train',
                                     transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=os.path.join(data_dir, 'val.json'), mode='val',
                                   transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                  pin_memory=True, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                pin_memory=True, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader
