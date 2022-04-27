import os
import os.path as osp
import argparse
import warnings

import wandb
from torch.optim import *
from tqdm import tqdm

from dataset import load_dataset
from loss import get_loss
from smp_model import build_model
from utils import *
from wandb_setup import wandb_init

warnings.filterwarnings('ignore')

class_labels = {
    0: "Backgroud",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/opt/ml/input/data')
    parser.add_argument('--config-dir', type=str, default='./config.yaml')
    parser.add_argument('--name', type=str, default="test")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--viz-log', type=int, default=20)
    parser.add_argument('--metric', action='store_true')
    parser.add_argument('--loss', action='store_true')
    parser.add_argument('--save-interval', default=5)
    parser.add_argument('--work_dir', type=str, default='./work_dirs',
                        help='the root dir to save logs and models about each experiment')
    arg = parser.parse_args()
    return arg


def main():
    args = get_parser()
    args = load_config(args)
    set_seed(args.seed)

    model, preprocessing_fn = build_model(args)
    train_loader, val_loader = load_dataset(args, preprocessing_fn)

    # Loss
    criterion = get_loss(args.criterion)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

    # Wandb init
    wandb_init(args)
    wandb.config = {
        "learning_rate": args.lr,
        "encoder": args.encoder,
        "epochs": args.epoch,
        "batch_size": args.batch_size
    }
    wandb.watch(model)

    device = args.device
    best_loss = 9999999.0
    best_score = 0.0
    
    for epoch in range(1, args.epoch + 1):
        model.train()
        train_loss, train_miou_score, train_accuracy = 0, 0, 0
        train_f1_score, train_recall, train_precision = 0, 0, 0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}] Train")
        for i, data in enumerate(pbar):
            image, mask = data
            image, mask = image.to(device), mask.to(device)
            output = model(image)

            optimizer.zero_grad()
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_miou_score += mIoU(output, mask)
            train_accuracy += pixel_accuracy(output, mask)
            f1_score, recall, precision = get_metrics(output, mask)
            train_f1_score += f1_score.item()
            train_recall += recall.item()
            train_precision += precision.item()
            pbar.set_postfix(
                Train_Loss=f" {train_loss/(i+1):.3f}",
                Train_Iou=f" {train_miou_score/(i+1):.3f}",
                Train_Acc=f" {train_accuracy/(i+1):.3f}",
            )
        wandb.log({
            'train/loss': train_loss/len(train_loader),
            'train/miou_score': train_miou_score/len(train_loader),
            'train/pixel_accuracy': train_accuracy/len(train_loader),
            'train/f1_score': train_f1_score/len(train_loader),
            'train/recall': train_recall/len(train_loader),
            'train/precision': train_precision/len(train_loader),
            'learning_rate': scheduler.get_lr()[0],
        }, commit=False)


        scheduler.step()
        val_loss, val_miou_score, val_accuracy = 0, 0, 0
        val_f1_score, val_recall, val_precision = 0, 0, 0
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"[Epoch {epoch}] Valid")
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(val_pbar):
                image, mask = data
                image, mask = image.to(device), mask.to(device)
                output = model(image)

                loss = criterion(output, mask)
                val_loss += loss.item()
                val_miou_score += mIoU(output, mask)
                val_accuracy += pixel_accuracy(output, mask)
                f1_score, recall, precision = get_metrics(output, mask)
                val_f1_score += f1_score.item()
                val_recall += recall.item()
                val_precision += precision.item()

                val_pbar.set_postfix(
                    Valid_Loss=f" {val_loss / (i + 1):.3f}",
                    Valid_Iou=f" {val_miou_score / (i + 1):.3f}",
                    Valid_Acc=f" {val_accuracy / (i + 1):.3f}",
                )
                output = torch.argmax(output, dim=1).detach().cpu().numpy()
                if args.viz_log == i:
                    wandb.log({
                        'visualize': wandb.Image(
                            image[0, :, :, :],
                            masks={
                                "predictions": {
                                    "mask_data": output[0, :, :],
                                    "class_labels": class_labels
                                },
                                "ground_truth": {
                                    "mask_data": mask[0, :, :].detach().cpu().numpy(),
                                    "class_labels": class_labels
                                }
                            }
                        )
                    }, commit=False)
            wandb.log({
                'val/loss': val_loss/len(val_loader),
                'val/miou_score': val_miou_score/len(val_loader),
                'val/pixel_accuracy': val_accuracy/len(val_loader),
                'val/f1_score': val_f1_score/len(val_loader),
                'val/recall': val_recall/len(val_loader),
                'val/precision': val_precision/len(val_loader),
            })
        # save_model
        if args.metric:
            if best_score < val_miou_score:
                best_score = val_miou_score
                ckpt_path = os.path.join(args.work_dir_exp, 'best_miou.pth')
                torch.save(model.state_dict(), ckpt_path)
        if not args.metric:
            if best_loss > val_loss:
                best_loss = val_loss
                ckpt_path = os.path.join(args.work_dir_exp, 'best_loss.pth')
                torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % args.save_interval == 0:
            ckpt_fpath = os.path.join(args.work_dir_exp, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

if __name__ == "__main__":
    main()
