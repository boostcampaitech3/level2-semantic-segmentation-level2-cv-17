import wandb

def wandb_login():
    wandb.login(key='your_key')

def wandb_init(args):
    wandb.init(
        project="semantic_segmentation_",
        entity="mg_generation",
        name='_'.join([args.work_dir_exp.split('/')[-1], args.encoder, args.decoder]),
        reinit=True,
        config = {
            "encoder": args.encoder,
            "encoder_weights": args.encoder_weights,
            "decoder": args.decoder,
            
            "criterion": args.criterion,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,

            "epochs": args.epoch,
            "learning_rate": args.lr,
            "batch_size": args.batch_size
        },
    )