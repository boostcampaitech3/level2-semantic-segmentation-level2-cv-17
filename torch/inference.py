import argparse

import pandas as pd
import albumentations as albu

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import CustomDataLoader, collate_fn
from smp_model import build_model
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--work-dir', type=str, default='./work_dirs')
    parser.add_argument('--config-dir', type=str, default='./config.yaml')

    parser.add_argument('--exp-name', type=str, default='exp_7')
    parser.add_argument('--model-name', type=str, default='best_miou')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    args = load_config(args)
    set_seed(args.seed)
    
    device = args.device
    
    model, preprocessing_fn = build_model(args)
    test_dataset = CustomDataLoader(data_dir=os.path.join(args.data_dir, 'test.json'), mode='test',
                                    preprocessing=preprocessing_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                 pin_memory=True, collate_fn=collate_fn)
    model_dir = os.path.join(args.work_dir_exp, f"{args.model_name}.pth")
    model.load_state_dict(torch.load(model_dir))
    model.to(device)

    model.eval()
    size = 256
    transform = albu.Compose([albu.Resize(size, size)])
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test")
    with torch.no_grad():
        for i, (images, image_infos) in enumerate(pbar):
            output = model(torch.stack(images).float().to(device))
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

    submission = pd.read_csv('../submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)
    submission.to_csv(os.path.join(args.work_dir_exp, f"{args.model_name}.csv"), index=False)

if __name__ == "__main__":
    main()
