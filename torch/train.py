import argparse
import warnings

import wandb

from tqdm import tqdm

from dataset import load_dataset
from loss import get_loss
from optimizer import get_optimizer
from scheduler import get_scheduler
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
    parser.add_argument('--save_interval', default=10)
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
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.scheduler)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epoch)

    wandb_login()
    wandb_init(args)
    wandb.watch(model)

    best_loss, best_loss_epoch = 9999999.0, 0
    best_score, best_score_epoch = 0.0, 0
    
    for epoch in range(1, args.epoch + 1):
        # train
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
        }, commit=False)
        

        # valid
        with torch.no_grad():
            val_loss, val_miou_score, val_accuracy = 0, 0, 0
            val_f1_score, val_recall, val_precision = 0, 0, 0

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
                            masks={"predictions": {"mask_data": output[0, :, :],
                                                   "class_labels": class_labels},
                                   "ground_truth": {"mask_data": masks[0, :, :].detach().cpu().numpy(),
                                                    "class_labels": class_labels}
                            }
                        )
                    }, commit=False)
            
            IoU_by_class = [
                {cls_name: round(IoU,4)} for IoU, cls_name in zip( IoU, list(class_labels.values()) )
            ]
            for i, cls_name in class_labels.items():
                if cls_name == 'Clothing':
                    wandb.log({f'cls/{i} {cls_name}': IoU_by_class[i][cls_name]}, commit=False)
                else:
                    wandb.log({f'cls/0{i} {cls_name}': IoU_by_class[i][cls_name]}, commit=False)
                
            wandb.log({
                'val/loss': val_loss / len(val_loader),
                'val/miou_score': val_miou_score / len(val_loader),
                'val/accuracy': val_accuracy / len(val_loader),
                'val/f1_score': val_f1_score / len(val_loader),
                'val/recall': val_recall / len(val_loader),
                'val/precision': val_precision / len(val_loader),
            }, commit=False)
        
        wandb.log({
            'learning_rate': scheduler.optimizer.param_groups[0]['lr']
        })

        # scheduler step
        if args.scheduler == 'reduce':
            scheduler.step(val_loss / len(val_loader))
        else:
            scheduler.step()

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
        
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.work_dir_exp, f'epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)

        if epoch == args.epoch:
            new_best_score_path = best_score_path[:-4] + f"_epoch{best_score_epoch}.pth"
            new_best_loss_path = best_loss_path[:-4] + f"_epoch{best_loss_epoch}.pth"
            os.rename(best_score_path, new_best_score_path)
            os.rename(best_loss_path, new_best_loss_path)


if __name__ == "__main__":
    main()
