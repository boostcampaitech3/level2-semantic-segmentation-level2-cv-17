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


def get_train_transform(args, preprocessing_fn):
    transform = []
    transform.append(A.Resize(512, 512))
    # transform.append(A.Normalize(mean=[107.0572, 112.4904, 117.9002], std=[55.0084, 52.941, 53.738], max_pixel_value=1.0))
    transform.append(A.Lambda(image=preprocessing_fn))
    transform.append(ToTensorV2())
    return args, A.Compose(transform)


def get_valid_transform(args, preprocessing_fn):
    transform = []
    transform.append(A.Resize(512, 512))
    # transform.append(A.Normalize(mean=[107.0572, 112.4904, 117.9002], std=[55.0084, 52.941, 53.738], max_pixel_value=1.0))
    transform.append(A.Lambda(image=preprocessing_fn))
    transform.append(ToTensorV2())
    return args, A.Compose(transform)


def get_test_transform(args, preprocessing_fn):
    transform = []
    transform.append(A.Resize(512, 512))
    # transform.append(A.Normalize(mean=[107.0572, 112.4904, 117.9002], std=[55.0084, 52.941, 53.738], max_pixel_value=1.0))
    transform.append(A.Lambda(image=preprocessing_fn))
    transform.append(ToTensorV2())
    return args, A.Compose(transform)
