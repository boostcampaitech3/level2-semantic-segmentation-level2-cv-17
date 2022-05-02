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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data') # do not change

    parser.add_argument('--work-dir', type=str, default='./work_dirs/exp0') # change this
    parser.add_argument('--src-config', type=str, default='train_config.yaml', help='Base config')
    parser.add_argument('--dst-config', type=str, default='test_config.yaml', help='Save config')
    parser.add_argument('--ckpt-name', type=str, default='best_miou_epoch1.pth') # change this
    parser.add_argument('--save-remark', type=str, default='', help='this will be added in csv file name')
    
    args = parser.parse_args()
    args.src_config = os.path.join(args.work_dir, args.src_config)
    args.dst_config = os.path.join(args.work_dir, args.dst_config)
    args.ckpt_name = os.path.join(args.work_dir, args.ckpt_name)
    return args


def main():
    args = get_parser()
    args = load_config(args)
    set_seed(args.seed)
    
    model, preprocessing_fn = build_model(args)
    args, test_transform = get_test_transform(args, preprocessing_fn)
    test_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'test.json'), mode='test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, pin_memory=True, collate_fn=collate_fn)
    model.load_state_dict(torch.load(args.ckpt_name))
    model.to(args.device)

    save_config(args)

    model.eval()
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test")
    with torch.no_grad():
        for idx, (images, image_infos) in enumerate(pbar):
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

    submission = pd.DataFrame(columns=['image_id', 'PredictionString'])
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())}, ignore_index=True)
    
    if args.save_remark != '':
        save_name = args.ckpt_name[:-4] + '_' + args.save_remark + '.csv'
    else:
        save_name = args.ckpt_name[:-4] + '.csv'
    submission.to_csv(save_name, index=False)


if __name__ == "__main__":
    main()
