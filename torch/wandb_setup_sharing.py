import wandb

api_key = ''
project_name = ''


def wandb_login():
    wandb.login(key=api_key)

def wandb_init(args):
    wandb.init(
        project=project_name,
        entity="mg_generation",
        name=get_wandb_run_name(args),
        tags=['torch'],
        reinit=True,
        config=args.__dict__,
    )

def get_wandb_run_name(args):
    wandb_name = [args.work_dir_exp.split('/')[-1], args.encoder, args.decoder]
    if args.wandb_remark != '':
        wandb_name.append(args.wandb_remark)
    return '_'.join(wandb_name)

def sweep_init(args):
    wandb.init(
        name=args.work_dir_exp.split('/')[-1],
        group=args.sweep_name,
        tags=['torch'],
        reinit=True,
    )

def get_sweep_id(sweep_config):
    sweep_id = wandb.sweep(
        project=project_name,
        entity="mg_generation",
        sweep=sweep_config,
    )
    return sweep_id

def get_sweep_config(args):
    # you must read this : https://docs.wandb.ai/guides/sweeps/configuration
    sweep_cfg = dict(
        name=args.sweep_name,
        method='grid',
        metric=dict(
            name='val/miou_score',
            goal='maximize'
        ))
    if sweep_cfg['method'] == 'grid':
        sweep_cfg['parameters'] = dict(
            classes=dict(values=[11]),
            num_worker=dict(values=[8]),

            fold=dict(values=[0,1,2,3,4,5,6,7,8,9]),
            lr=dict(values=[0.0001]),
            epoch=dict(values=[10]),
            batch_size=dict(values=[8]),

            decoder=dict(values=['FPN']),
            encoder=dict(values=['timm-efficientnet-b4']),
            encoder_depth=dict(values=[5]),
            encoder_weights=dict(values=['imagenet']),
            activation=dict(values=[None]),
            aux_params=dict(values=[None]),

            criterion=dict(values=['CE']),
            optimizer=dict(values=['Adam']),
            scheduler=dict(values=['multistep']),
        )
    elif sweep_cfg['method'] == 'bayes':
        sweep_cfg['parameters'] = dict(
            classes=dict(distribution='categorical', values=[11]),
            num_worker=dict(distribution='categorical', values=[8]),

            fold=dict(distribution='categorical', values=[0]),
            lr=dict(distribution='categorical', values=[0.0001]),
            epoch=dict(distribution='categorical', values=[1]),
            batch_size=dict(distribution='categorical', values=[8]),

            decoder=dict(distribution='categorical', values=['FPN']),
            encoder=dict(distribution='categorical', values=['timm-efficientnet-b4']),
            encoder_depth=dict(distribution='categorical', values=[5]),
            encoder_weights=dict(distribution='categorical', values=['imagenet']),
            activation=dict(distribution='categorical', values=[None]),
            aux_params=dict(distribution='categorical', values=[None]),

            criterion=dict(distribution='categorical', values=['dice']),
            optimizer=dict(distribution='categorical', values=['Adam']),
            scheduler=dict(distribution='categorical', values=['multistep']),
        )
    return sweep_cfg
