import albumentations as A
from albumentations.pytorch import ToTensorV2


# Segmentation 대회 추천 Augmentation
'''
A.HorizontalFlip(),
A.VerticalFlip(),
A.ChannelShuffle(),
A.GaussNoise(),
A.RandomBrightness(),
A.CLAHE(),
A.ColorJitter()     

A.ElasticTransform() ?
'''


def get_train_transform(preprocessing_fn):
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2(),
        ]
    )


def get_valid_transform(preprocessing_fn):
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2(),
        ]
    )


def get_test_transform(preprocessing_fn):
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2(),
        ]
    )