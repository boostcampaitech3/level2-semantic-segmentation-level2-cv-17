import argparse

import pandas as pd
import albumentations as A

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import CustomDataLoader, collate_fn
from smp_model import build_model
from utils import *
from transform import get_test_transform


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
    set_seed(args.seed)
    
    model, preprocessing_fn = build_model(args) # smp_model.py
    args, test_transform = get_test_transform(args, preprocessing_fn) # transform.py
    test_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'test.json'), mode='test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, pin_memory=True, collate_fn=collate_fn)
    model.load_state_dict(torch.load(args.ckpt_dir))
    model.to(args.device)
    
    # ex. pth : best_miou_epoch1.pth
    # -> yaml : test_config_best_miou_epoch1_{args.save_remark}.yaml
    save_config_dir = maybe_apply_remark(args.dst_config_dir, args.ckpt_name.split('.')[0], '.yaml')
    save_config_dir = maybe_apply_remark(save_config_dir, args.save_remark, '.yaml')
    save_config(args, save_config_dir)

    model.eval()
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test")
    with torch.no_grad():
        for idx, (images, image_infos) in enumerate(pbar):
            if args.aux_params:
                output, output_label = model(torch.stack(images).float().to(args.device))
            else:
                output = model(torch.stack(images).float().to(args.device))
            oms = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()

            temp_mask = []
            images = torch.stack(images).float().detach().cpu().numpy()
            for img, mask in zip(images, oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
        
        file_names = [y for x in file_name_list for y in x]

    # sample_submission is not needed
    submission = pd.DataFrame(columns=['image_id', 'PredictionString'])
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())}, ignore_index=True)
    
    # ex. pth : best_miou_epoch1.pth
    # ->  csv : best_miou_epoch1_{args.save_remark}.csv
    save_csv_dir = maybe_apply_remark(args.ckpt_dir, args.save_remark, '.csv')
    submission.to_csv(save_csv_dir, index=False)


if __name__ == "__main__":
    main()
