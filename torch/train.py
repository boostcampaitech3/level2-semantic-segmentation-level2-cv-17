import argparse
import warnings

import wandb

from tqdm import tqdm
from torch.optim import *

from dataset import load_dataset
from loss import get_loss
from smp_model import build_model
from utils import *
from wandb_setup import wandb_login, wandb_init

warnings.filterwarnings('ignore')

class_labels = {
    0: "Background",
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
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_dir', default='/opt/ml/input/data')
    parser.add_argument('--work_dir', type=str, default='./work_dirs')
    parser.add_argument('--config_dir', type=str, default='./config.yaml')
    
    parser.add_argument('--viz_log', type=int, default=20)
    parser.add_argument('--check_train_data', action='store_true', default=False)
    parser.add_argument('--save_interval', default=5)
    arg = parser.parse_args()
    return arg


def main():
    args = get_parser()
    args = load_config(args)
    set_seed(args.seed)

    model, preprocessing_fn = build_model(args)
    train_loader, val_loader = load_dataset(args, preprocessing_fn)
    model.to(args.device)
    
    criterion = get_loss(args.criterion)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

    wandb_login()
    wandb_init(args)
    wandb.config = {
        "learning_rate": args.lr,
        "encoder": args.encoder,
        "epochs": args.epoch,
        "batch_size": args.batch_size
    }
    wandb.watch(model)

    best_loss, best_loss_epoch = 9999999.0, 0
    best_score, best_score_epoch = 0.0, 0
    
    for epoch in range(1, args.epoch + 1):
        train_loss, train_miou_score, train_accuracy = 0, 0, 0
        train_f1_score, train_recall, train_precision = 0, 0, 0

        model.train()
        hist = np.zeros((args.classes, args.classes))
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}] Train")
        for idx, data in enumerate(pbar):
            images, masks = data
            images = torch.stack(images).float().to(args.device)
            masks = torch.stack(masks).long().to(args.device)
            output = model(images)

            if args.check_train_data:
                if idx == 0:
                    batch_train_d = []
                    for train_img in images:
                        batch_train_d.append(wandb.Image(train_img))
                    wandb.log({'train_image':batch_train_d}, commit=False)

            optimizer.zero_grad()
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            hist = add_hist(hist, masks, output, n_class=args.classes)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            train_miou_score += mIoU
            train_accuracy += acc

            f1_score, recall, precision = get_metrics(output, masks)
            train_f1_score += f1_score.item()
            train_recall += recall.item()
            train_precision += precision.item()
            pbar.set_postfix(
                Train_Loss=f" {train_loss / (idx+1):.3f}",
                Train_Iou=f" {train_miou_score / (idx+1):.3f}",
                Train_Acc=f" {train_accuracy / (idx+1):.3f}",
            )
            
        wandb.log({
            'train/loss': train_loss / len(train_loader),
            'train/miou_score': train_miou_score / len(train_loader),
            'train/accuracy': train_accuracy / len(train_loader),
            'train/f1_score': train_f1_score / len(train_loader),
            'train/recall': train_recall / len(train_loader),
            'train/precision': train_precision / len(train_loader),
            'learning_rate': scheduler.get_lr()[0],
        }, commit=False)

        scheduler.step()

        val_loss, val_miou_score, val_accuracy = 0, 0, 0
        val_f1_score, val_recall, val_precision = 0, 0, 0
        
        with torch.no_grad():
            model.eval()
            hist = np.zeros((args.classes, args.classes))
            val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"[Epoch {epoch}] Valid")
            for idx, data in enumerate(val_pbar):
                images, masks = data
                images = torch.stack(images).float().to(args.device)
                masks = torch.stack(masks).long().to(args.device)
                output = model(images)

                loss = criterion(output, masks)
                val_loss += loss.item()
                
                hist = add_hist(hist, masks, output, n_class=args.classes)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                val_miou_score += mIoU
                val_accuracy += acc
                
                f1_score, recall, precision = get_metrics(output, masks)
                val_f1_score += f1_score.item()
                val_recall += recall.item()
                val_precision += precision.item()
                val_pbar.set_postfix(
                    Valid_Loss=f" {val_loss / (idx+1):.3f}",
                    Valid_Iou=f" {val_miou_score / (idx+1):.3f}",
                    Valid_Acc=f" {val_accuracy / (idx+1):.3f}",
                )

                output = torch.argmax(output, dim=1).detach().cpu().numpy()
                if args.viz_log == idx:
                    wandb.log({
                        'visualize': wandb.Image(
                            images[0, :, :, :],
                            masks={
                                "predictions": {
                                    "mask_data": output[0, :, :],
                                    "class_labels": class_labels
                                },
                                "ground_truth": {
                                    "mask_data": masks[0, :, :].detach().cpu().numpy(),
                                    "class_labels": class_labels
                                }
                            }
                        )
                    }, commit=False)
            
            IoU_by_class = [
                {class_name:round(IoU,4)} for IoU,class_name in zip( IoU, list(class_labels.values()) )
            ]
            
            wandb.log({
                'val/loss': val_loss / len(val_loader),
                'val/miou_score': val_miou_score / len(val_loader),
                'val/accuracy': val_accuracy / len(val_loader),
                'val/f1_score': val_f1_score / len(val_loader),
                'val/recall': val_recall / len(val_loader),
                'val/precision': val_precision / len(val_loader),
                'cls/0 Background': IoU_by_class[0]['Background'],
                'cls/1 General trash': IoU_by_class[1]['General trash'],
                'cls/2 Paper': IoU_by_class[2]['Paper'],
                'cls/3 Paper pack': IoU_by_class[3]['Paper pack'],
                'cls/4 Metal': IoU_by_class[4]['Metal'],
                'cls/5 Glass': IoU_by_class[5]['Glass'],
                'cls/6 Plastic': IoU_by_class[6]['Plastic'],
                'cls/7 Styrofoam': IoU_by_class[7]['Styrofoam'],
                'cls/8 Plastic bag': IoU_by_class[8]['Plastic bag'],
                'cls/9 Battery': IoU_by_class[9]['Battery'],
                'cls/10 Clothing': IoU_by_class[10]['Clothing'],
            })
            
        # save_model
        if best_score < val_miou_score:
            best_score = val_miou_score
            best_score_epoch = epoch
            best_score_path = os.path.join(args.work_dir_exp, 'best_miou.pth')
            torch.save(model.state_dict(), best_score_path)
        if best_loss > val_loss:
            best_loss = val_loss
            best_loss_epoch = epoch
            best_loss_path = os.path.join(args.work_dir_exp, 'best_loss.pth')
            torch.save(model.state_dict(), best_loss_path)
        if (epoch + 1) % args.save_interval == 0:
            ckpt_fpath = os.path.join(args.work_dir_exp, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        if epoch == args.epoch:
            new_best_score_path = best_score_path[:-4] + f"_epoch{best_score_epoch}.pth"
            new_best_loss_path = best_loss_path[:-4] + f"_epoch{best_loss_epoch}.pth"
            os.rename(best_score_path, new_best_score_path)
            os.rename(best_loss_path, new_best_loss_path)


if __name__ == "__main__":
    main()
