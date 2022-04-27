import wandb

def wandb_login():
    wandb.login(key='your_key')

def wandb_init(args):
    wandb.init(
        project="project_name",
        entity="entity_name",
        name='_'.join([args.work_dir_exp.split('/')[-1], args.encoder, args.decoder]),
        reinit=True)