import argparse
import warnings

import wandb
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from functools import partial

from dataset import load_dataset
from loss import get_loss
from optimizer import get_optimizer
from scheduler import get_scheduler
from smp_model import build_model
from utils import *
from wandb_setup import *

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 15,15 # wandb image 크기 조절

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
    parser.add_argument('--mode', type=str, default="train") # do not change
    parser.add_argument('--seed', type=int, default=42) # maybe do not change

    parser.add_argument('--data-dir', '-d', type=str, default='/opt/ml/input/data/')
    parser.add_argument('--add-train', '-a', type=str, nargs='+', default=['/opt/ml/input/data/leak.json',
                                                                           '/opt/ml/input/data/val_clean.json',
                                                                        #    '/opt/ml/input/data/hyo_7941.json'
                                                                           ])
    parser.add_argument('--work-dir', type=str, default='./work_dirs') # do not change
    parser.add_argument('--src-config', type=str, default='config.yaml', help='Base config') # do not change
    parser.add_argument('--dst-config', type=str, default='train_config.yaml', help='Save config') # do not change
    
    parser.add_argument('--save-interval', '-i', type=int, default=10, help='.pth save interval')
    parser.add_argument('--train-image', '-t', action='store_true', default=True, help='if you want to see augmented image')
    parser.add_argument('--valid-image', '-v', action='store_true', default=True, help='if you want to see evaluation image')
    
    parser.add_argument('--wandb-remark', '-r', type=str, default='', help='this will be added in wandb run name')
    parser.add_argument('--sweep', action='store_true', default=False, help='sweep True')
    parser.add_argument('--sweep-name', '-n', type=str, default='sweep', help='this will be sweep name and also group name')

    args = parser.parse_args()
    args.src_config_dir = os.path.join(os.getcwd(), args.src_config) # maybe do not change
    return args


def do_train(args, model, train_loader, val_loader, optimizer, criterion, cls_criterion, sub_criterion, scheduler):
    model.to(args.device)
    wandb.watch(model)

    best_loss, best_loss_epoch = 9999999.0, 0
    best_score, best_score_epoch = 0.0, 0

    for epoch in range(1, args.epoch + 1):
        # train
        train_loss, train_miou_score, train_accuracy = 0, 0, 0
        train_mask_loss, train_cls_loss, train_sub_loss = 0, 0, 0
        train_f1_score, train_recall, train_precision = 0, 0, 0

        model.train()
        hist = np.zeros((args.classes, args.classes))
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}] Train")
        for idx, (images, masks, labels) in enumerate(pbar):
            images = torch.stack(images).float().to(args.device)
            masks = torch.stack(masks).long().to(args.device)
            labels = torch.stack(labels).float().to(args.device)
            if args.aux_params:
                output, output_label = model(images)
            else:
                output = model(images)

            optimizer.zero_grad()
            loss = criterion(output, masks)
            train_mask_loss += loss.item()
            total_loss = loss
            if args.aux_params:
                loss_label = cls_criterion(output_label, labels)
                train_cls_loss += loss_label.item()
                total_loss += loss_label
            if args.sub_criterion:
                loss_sub = sub_criterion(output, masks)
                train_sub_loss += loss_sub.item()
                total_loss += loss_sub
            train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
            
            hist = add_hist(hist, masks, output, n_class=args.classes)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            train_miou_score += mIoU
            train_accuracy += acc

            f1_score, recall, precision = get_metrics(output, masks)
            train_f1_score += f1_score.item()
            train_recall += recall.item()
            train_precision += precision.item()
            pbar.set_postfix(
                Train_Loss=f" {train_loss / (idx+1):.3f}", Train_Iou=f" {train_miou_score / (idx+1):.3f}", Train_Acc=f" {train_accuracy / (idx+1):.3f}",
            )
            
            if args.train_image:
                if idx in [0]:
                    batch_img = []
                    for train_img in images.detach().cpu().numpy():
                        train_img = np.transpose(train_img, (1,2,0))
                        if args.norm:
                            train_img = (train_img * args.norm_std) + args.norm_mean
                            train_img = np.clip(train_img*255, 0, 255)
                        batch_img.append(wandb.Image(train_img))
                    wandb.log({'train_image':batch_img}, commit=False)
            
        wandb.log({
            'train/loss': train_loss / len(train_loader), 'train/miou_score': train_miou_score / len(train_loader), 'train/accuracy': train_accuracy / len(train_loader),
            'train/f1_score': train_f1_score / len(train_loader), 'train/recall': train_recall / len(train_loader), 'train/precision': train_precision / len(train_loader),
            'train/mask_loss': train_mask_loss / len(train_loader), 'train/cls_loss': train_cls_loss / len(train_loader), 'train/sub_loss': train_sub_loss / len(train_loader),
        }, commit=False)
        

        # valid
        with torch.no_grad():
            val_loss, val_miou_score, val_accuracy = 0, 0, 0
            val_mask_loss, val_cls_loss, val_sub_loss = 0, 0, 0
            val_f1_score, val_recall, val_precision = 0, 0, 0
            
            model.eval()
            hist = np.zeros((args.classes, args.classes))
            val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"[Epoch {epoch}] Valid")
            for idx, (images, masks, labels) in enumerate(val_pbar):
                images = torch.stack(images).float().to(args.device)
                masks = torch.stack(masks).long().to(args.device)
                labels = torch.stack(labels).float().to(args.device)
                if args.aux_params:
                    output, output_label = model(images)
                else:
                    output = model(images)

                loss = criterion(output, masks)
                val_mask_loss += loss.item()
                total_val_loss = loss
                if args.aux_params:
                    loss_label = cls_criterion(output_label, labels)
                    val_cls_loss += loss_label.item()
                    total_val_loss += loss_label
                if args.sub_criterion:
                    loss_sub = sub_criterion(output, masks)
                    val_sub_loss += loss_sub.item()
                    total_val_loss += loss_sub
                val_loss += total_val_loss.item()
                             
                hist = add_hist(hist, masks, output, n_class=args.classes)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                val_miou_score += mIoU
                val_accuracy += acc
                
                f1_score, recall, precision = get_metrics(output, masks)
                val_f1_score += f1_score.item()
                val_recall += recall.item()
                val_precision += precision.item()
                val_pbar.set_postfix(
                    Valid_Loss=f" {val_loss / (idx+1):.3f}", Valid_Iou=f" {val_miou_score / (idx+1):.3f}", Valid_Acc=f" {val_accuracy / (idx+1):.3f}",
                )

                if args.valid_image:
                    if idx in [19,23,25,35,42,47,58,63]:
                        valid_img = images[0, :, :, :].detach().cpu().numpy()
                        valid_img = np.transpose(valid_img, (1,2,0))
                        if args.norm:
                            valid_img = (valid_img * args.norm_std) + args.norm_mean
                            valid_img = np.clip(valid_img*255, 0, 255)
                        output_argmax = torch.argmax(output, dim=1).detach().cpu().numpy()
                        wandb.log({
                            f'visualize_{str(idx).zfill(2)}': wandb.Image(
                                valid_img,
                                masks={"predictions": {"mask_data": output_argmax[0, :, :], "class_labels": class_labels},
                                       "ground_truth": {"mask_data": masks[0, :, :].detach().cpu().numpy(), "class_labels": class_labels}}
                            )}, commit=False)
            
            hist = np.array(hist).astype('float')
            with np.errstate(divide='ignore', invalid='ignore'): norm_hist = hist / np.array([hist.sum(axis=1)]).T
            df_hist = pd.DataFrame(norm_hist, index=list(class_labels.values()), columns=list(class_labels.values()))
            cm = sns.heatmap(df_hist, annot=True, linewidths=.5, annot_kws={"size": 10})
            plt.title(f'confusion_matrix_true_epoch{epoch}')
            plt.xlabel('Pred')
            plt.ylabel('True')
            wandb.log({'confusion_matrix_true': wandb.Image(cm)}, commit=False)
            plt.clf()
            with np.errstate(divide='ignore', invalid='ignore'): norm_hist = hist / hist.sum(axis=0)
            df_hist = pd.DataFrame(norm_hist, index=list(class_labels.values()), columns=list(class_labels.values()))
            cm = sns.heatmap(df_hist, annot=True, linewidths=.5, annot_kws={"size": 10})
            plt.title(f'confusion_matrix_pred_epoch{epoch}')
            plt.xlabel('Pred')
            plt.ylabel('True')
            wandb.log({'confusion_matrix_pred': wandb.Image(cm)}, commit=False)
            plt.clf()

            IoU_by_class = [
                {cls_name: round(IoU,4)} for IoU, cls_name in zip( IoU, list(class_labels.values()) )
            ]
            for i, cls_name in class_labels.items():
                wandb.log({f'cls/{str(i).zfill(2)} {cls_name}': IoU_by_class[i][cls_name]}, commit=False)
            
            wandb.log({
                'val/loss': val_loss / len(val_loader), 'val/miou_score': val_miou_score / len(val_loader), 'val/accuracy': val_accuracy / len(val_loader),
                'val/f1_score': val_f1_score / len(val_loader), 'val/recall': val_recall / len(val_loader), 'val/precision': val_precision / len(val_loader),
                'val/mask_loss': val_mask_loss / len(val_loader), 'val/cls_loss': val_cls_loss / len(val_loader), 'val/sub_loss': val_sub_loss / len(val_loader),
            }, commit=False)
        
        wandb.log({'learning_rate': scheduler.optimizer.param_groups[0]['lr']})

        # scheduler step
        if args.scheduler == 'reduce': scheduler.step(val_loss / len(val_loader))
        else: scheduler.step()

        # save model
        if best_score < val_miou_score:
            best_score, best_score_epoch = val_miou_score, epoch
            best_score_path = os.path.join(args.work_dir_exp, 'best_miou.pth')
            torch.save(model.state_dict(), best_score_path)
        # if best_loss > val_loss:
        #     best_loss, best_loss_epoch = val_loss, epoch
        #     best_loss_path = os.path.join(args.work_dir_exp, 'best_loss.pth')
        #     torch.save(model.state_dict(), best_loss_path)
        # and also every save_interval
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.work_dir_exp, f'epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
        # best_loss, best_miou will get epoch info
        if epoch == args.epoch:
            new_best_score_path = best_score_path[:-4] + f"_epoch{best_score_epoch}.pth"
            # new_best_loss_path = best_loss_path[:-4] + f"_epoch{best_loss_epoch}.pth"
            os.rename(best_score_path, new_best_score_path)
            # os.rename(best_loss_path, new_best_loss_path)
            if args.epoch < args.save_interval:
                ckpt_path = os.path.join(args.work_dir_exp, f'epoch{epoch}.pth')
                torch.save(model.state_dict(), ckpt_path)


def main(args):
    set_seed(args.seed)
    args.work_dir_exp = get_exp_dir(args.work_dir)
    args.dst_config_dir = os.path.join(args.work_dir_exp, args.dst_config)

    if args.sweep:
        sweep_init(args)
        args = concat_config(args, wandb.config) # args + wandb.config
    else:
        base_config = load_config(args)
        args = concat_config(args, base_config) # args + base_config

    # all changes saved in args
    # sweep parmas must not be changed
    args, (model, preprocessing_fn) = build_model(args) # smp_model.py
    if args.ckpt_dir: model.load_state_dict(torch.load(args.ckpt_dir))
    args, (train_loader, val_loader) = load_dataset(args, preprocessing_fn) # datasat.py
    args, criterion, cls_criterion, sub_criterion = get_loss(args) # loss.py
    args, optimizer = get_optimizer(args, model.parameters()) # optimizer.py
    args, scheduler = get_scheduler(args, optimizer) # scheduler.py
    
    # args logged on wandb config
    if args.sweep:
        wandb.run.name = get_wandb_run_name(args)
        wandb.config = args.__dict__
    else:
        wandb_init(args)
    save_config(args, args.dst_config_dir) # args saved on args.dst_config_dir

    do_train(args, model, train_loader, val_loader, optimizer, criterion, cls_criterion, sub_criterion, scheduler)
    if args.sweep: wandb.finish()


if __name__ == "__main__":
    wandb_login()
    args = get_parser()
    if args.sweep:
        sweep_config = get_sweep_config(args)
        sweep_id = get_sweep_id(sweep_config)
        wandb.agent(sweep_id, function=partial(main, args))
    else:
        main(args)