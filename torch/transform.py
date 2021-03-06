import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
'''
A.HorizontalFlip()
A.VerticalFlip()
A.ChannelShuffle()
A.GaussNoise()
A.RandomBrightness()
A.CLAHE()
A.ColorJitter()
A.ElasticTransform()
'''

'''
train_all.json : mean=[0.46009143 0.43957698 0.41827274] std=[0.21060736 0.20755924 0.21633709]
test.json : mean=[0.46254247 0.43993051 0.41837708] std=[0.21567757 0.21132673 0.21795963]
leak.json : mean=[0.47148072 0.45122328 0.43063243] std=[0.19556618 0.19598319 0.20601789]
train_all.json + test.json : mean=[0.46131695, 0.43975374, 0.41832491] std=[0.21314246, 0.20944298, 0.21714836]
train_all.json + test.json + leak.json : mean=[0.46470487, 0.44357692, 0.42242742] std=[0.2072837, 0.20495638, 0.2134382]
'''

def get_train_transform(args, preprocessing_fn):
    args.norm = True
    args.norm_mean = [0.46470487, 0.44357692, 0.42242742]
    args.norm_std = [0.2072837, 0.20495638, 0.2134382]
    
    transform = [
        A.OneOf([
            A.CropNonEmptyMaskIfExists(384, 384),
        ], p=0.5),
        A.Resize(384, 384, interpolation=cv2.INTER_AREA),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.67),
        A.OneOf([
            A.ShiftScaleRotate(),
            A.RandomRotate90(),
        ], p=0.67),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
        ], p=0.67),
        A.OneOf([
            A.Emboss(),
            A.IAAEmboss(),
            A.Sharpen(),
        ], p=0.75),
        A.Normalize(mean=args.norm_mean, std=args.norm_std, max_pixel_value=1.0),
        ToTensorV2()
    ]
    return args, A.Compose(transform)


def get_valid_transform(args, preprocessing_fn):
    transform = []
    transform.append(A.Resize(512, 512))
    transform.append(A.Normalize(mean=args.norm_mean, std=args.norm_std, max_pixel_value=1.0))
    transform.append(ToTensorV2())
    return args, A.Compose(transform)
