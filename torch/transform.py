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

Elastic Transform??

'''

# 여기서 각자 원하는 augmentation 조합을 만들면 됩니다.
# 나중에 argparser로 관리도 가능할 듯

def train_transform(preprocessing_fn):
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2(),
      
        ]
    )

def valid_transform(preprocessing_fn):
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Lambda(image=preprocessing_fn),
            ToTensorV2(),
        ]
    )

