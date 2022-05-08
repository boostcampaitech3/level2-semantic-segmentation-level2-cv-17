import argparse

import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import CustomDataLoader, collate_fn
from smp_model import build_model
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="test") # do not change
    parser.add_argument('--seed', type=int, default=42) # maybe do not change

    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data') # do not change
    parser.add_argument('--work-dir', type=str, default='./work_dirs') # do not change
    parser.add_argument('--work-dir-exp', '-e', type=str, default='./work_dirs/exp14')
    parser.add_argument('--src-config', type=str, default='train_config.yaml', help='Base config') # maybe do not change
    parser.add_argument('--dst-config', type=str, default='test_config.yaml', help='Save config') # maybe do not change

    parser.add_argument('--ckpt-name', '-c', type=str, default='best_miou_epoch1.pth')
    parser.add_argument('--save-remark', '-r', type=str, default='', help='this will be added in csv, yaml name')

    args = parser.parse_args()
    args.src_config_dir = os.path.join(args.work_dir_exp, args.src_config)
    args.dst_config_dir = os.path.join(args.work_dir_exp, args.dst_config)
    args.ckpt_dir = os.path.join(args.work_dir_exp, args.ckpt_name)
    return args


def main():
    args = get_parser()
    config = load_config(args)
    args = concat_config(args, config)
    
    model, preprocessing_fn = build_model(args) # smp_model.py
    model.load_state_dict(torch.load(args.ckpt_dir))
    model.to(args.device)
    model.eval()

    args.TTA_flip_list = ['NoOp', 'HorizontalFlip', 'VerticalFlip']
    args.TTA_size_list = [512]

    # ex. pth : best_miou_epoch1.pth
    # -> yaml : test_config_best_miou_epoch1_{args.save_remark}.yaml
    save_config_dir = maybe_apply_remark(args.dst_config_dir, args.ckpt_name.split('.')[0], '.yaml')
    save_config_dir = maybe_apply_remark(save_config_dir, args.save_remark, '.yaml')
    save_config(args, save_config_dir)

    TTA_preds_array = []
    for tta_flip in args.TTA_flip_list:
        for tta_size in args.TTA_size_list:
            set_seed(args.seed)

            test_transform = []
            test_transform.append(A.Resize(tta_size, tta_size))
            test_transform.append(getattr(A, tta_flip)(p=1.0))
            test_transform.append(A.PadIfNeeded(512,512, border_mode=cv2.BORDER_CONSTANT))
            test_transform.append(A.Normalize(mean=args.norm_mean, std=args.norm_std, max_pixel_value=1.0))
            test_transform.append(ToTensorV2())
            test_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'test.json'), mode='test', transform=A.Compose(test_transform))
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, pin_memory=True, collate_fn=collate_fn)

            with torch.no_grad():
                file_name_list = []
                preds_array = np.empty((0, 256*256), dtype=np.long)
                pbar = tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test")
                
                for idx, (images, image_infos) in enumerate(pbar):
                    if args.aux_params:
                        output, output_label = model(torch.stack(images).float().to(args.device))
                    else:
                        output = model(torch.stack(images).float().to(args.device))
                    oms = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()

                    sub_transform = []
                    sub_transform.append(getattr(A, tta_flip)(p=1.0))
                    sub_transform.append(A.Resize(256, 256, interpolation=cv2.INTER_AREA)) # 이미지 축소 시, 보통 cv2.INTER_AREA 사용
                    
                    images = torch.stack(images).float().detach().cpu().numpy()
                    pad_value = int((512 - tta_size)/2)
                    images = images[:, :, pad_value:512-pad_value, pad_value:512-pad_value]
                    oms = oms[:, pad_value:512-pad_value, pad_value:512-pad_value]

                    temp_mask = []
                    for img, mask in zip(images, oms):
                        transformed = A.Compose(sub_transform)(image=img, mask=mask)
                        mask = transformed['mask']
                        temp_mask.append(mask)
                    oms = np.array(temp_mask)

                    oms = oms.reshape([oms.shape[0], 256*256]).astype(int)
                    preds_array = np.vstack((preds_array, oms))
                    file_name_list.append([i['file_name'] for i in image_infos])
                
                file_names = [y for x in file_name_list for y in x]

            TTA_preds_array.append(preds_array)

    sub_array = np.empty((0, 256*256), dtype=np.long)
    
    # TTA 경우의 수에 맞게끔 바꿔주시면 됩니다.
    if len(TTA_preds_array) == 2:
        pred1, pred2 = TTA_preds_array
        for p1,p2 in zip(pred1, pred2):
            tta_list = [p1.tolist(), p2.tolist()]
            count_list = max(tta_list, key=tta_list.count)
            sub_array = np.vstack((sub_array, count_list))
    elif len(TTA_preds_array) == 4:
        pred1, pred2, pred3, pred4 = TTA_preds_array
        for p1,p2,p3,p4 in zip(pred1, pred2, pred3, pred4):
            tta_list = [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()]
            count_list = max(tta_list, key=tta_list.count)
            sub_array = np.vstack((sub_array, count_list))
    elif len(TTA_preds_array) == 9:
        pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9 = TTA_preds_array
        for p1,p2,p3,p4,p5,p6,p7,p8,p9 in zip(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9):
            tta_list = [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist(), p5.tolist(), p6.tolist(), p7.tolist(), p8.tolist(), p9.tolist()]
            count_list = max(tta_list, key=tta_list.count)
            sub_array = np.vstack((sub_array, count_list))

    submission = pd.DataFrame(columns=['image_id', 'PredictionString'])
    for file_name, string in zip(file_names, sub_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())}, ignore_index=True)

    # ex. pth : best_miou_epoch1.pth
    # ->  csv : best_miou_epoch1_{args.save_remark}.csv
    save_csv_dir = maybe_apply_remark(args.ckpt_dir, args.save_remark, '.csv')
    submission.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    main()
