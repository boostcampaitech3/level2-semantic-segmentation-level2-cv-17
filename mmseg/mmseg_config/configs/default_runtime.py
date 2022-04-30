checkpoint_config = dict(interval=3, max_keep_ckpts = 1)

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='semantic_segmentation_hyo',
                entity = 'mg_generation',
                group = 'mmsegmentation',
                # config = None,
                # job_type = f'Fold{fold}',
                reinit = True
            ),
            )
    ])

# # yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
cudnn_benchmark = True # don't change to False