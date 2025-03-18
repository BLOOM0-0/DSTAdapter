_base_ = [
    '../../_base_/default_runtime_on_local.py'
]


# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='DST_Adapter',
        pretrained='tmp',
        input_resolution=224,
        patch_size=14,
        num_frames=8,
        width=1024,
        layers=24,
        heads=16,
        drop_path_rate=0.1,
        adapter_scale=0.5),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=263,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),

    data_preprocessor=dict(                     # 数据预处理器的配置
        type='ActionDataPreprocessor',          # 数据预处理器的名称
        mean=[122.769, 116.74, 104.04],         # 不同通道的均值用于归一化
        std=[68.493, 66.63, 70.321], # 注意clip和imagenet的不一样  # 不同通道的标准差用于归一化
        format_shape='NCTHW'))  # 最终图像形状的格式


# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/include/'+'videos'
data_root_val = data_root
ann_file_train = f'data/include/train_list.txt'
ann_file_val = f'data/include/val_list.txt'
ann_file_test = f'data/include/test_list.txt'



file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit'),
    # dict(type='SampleFrames', clip_len=32, frame_interval=4, num_clips=1),
    dict(type='UniformSample', clip_len=8),     # 将视频分为8个等长片段，每个片段1帧，共取8帧
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    # dict(type='SampleFrames', clip_len=32, frame_interval=4, num_clips=1),
    dict(type='UniformSample', clip_len=8),  # 将视频分为8个等长片段，每个片段1帧，共取8帧
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    # dict(type='SampleFrames', clip_len=32, frame_interval=4, num_clips=1),
    dict(type='UniformSample', clip_len=8),  # 将视频分为8个等长片段，每个片段1帧，共取8帧
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,       # 训练时每个单个 GPU 的批量大小
    num_workers=8,      # 训练时每个单个 GPU 的数据预取进程数
    persistent_workers=True,        # 如果为 `True`，则数据加载器在一个 epoch 结束后不会关闭工作进程，这可以加速训练速度
    sampler=dict(type='DefaultSampler', shuffle=True),   # 支持分布式和非分布式训练的 DefaultSampler
    dataset=dict(                          # 训练数据集的配置
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        # filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        # filename_tmpl='img_{:05}.jpg',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        # filename_tmpl='img_{:05}.jpg',
        pipeline=test_pipeline,
        test_mode=True))


train_cfg = dict(                       # 训练循环的配置
    type='EpochBasedTrainLoop',         # 训练循环的名称
    max_epochs=100,                      # 总的训练周期数
    val_begin=1,                        # 开始验证的训练周期
    dynamic_intervals=[(1, 5), (90, 1)])    # 验证间隔，从1开始每5轮验证一次，从20开始，一轮验证一次
val_cfg = dict(type='ValLoop')          # 验证循环的配置
test_cfg = dict(type='TestLoop')
val_evaluator = dict(type='AccMetric')  # 验证评估器的配置
test_evaluator = val_evaluator          # 测试评估器的配置

# optimizer
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',       # 优化器包装器的名称，切换到 AmpOptimWrapper 可以启用混合精度训练
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    paramwise_cfg=dict(class_embedding=dict(decay_mult=0.),
                        positional_embedding=dict(decay_mult=0.),
                        temporal_embedding=dict(decay_mult=0.),
                        absolute_pos_embed=dict(decay_mult=0.),
                        ln_1=dict(decay_mult=0.),
                        ln_2=dict(decay_mult=0.),
                        ln_pre=dict(decay_mult=0.),
                        ln_post=dict(decay_mult=0.)
                                    ))

# learning policy
param_scheduler = [                 # 更新优化器参数的学习率测率，支持字典或列表
    # dict(
    #     type='LinearLR',
    #     start_factor=0.1,
    #     by_epoch=True,
    #     begin=0,
    #     end=5,
    #     convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=100)
    # dict(type='MultiStepLR',  # 达到一个里程碑时衰减学习率
    #         begin=0,  # 开始更新学习率的步骤
    #         end=100,  # 结束更新学习率的步骤
    #         by_epoch=True,  # 是否按 epoch 更新学习率
    #         milestones=[60, 70, 90],  # 衰减学习率的步骤
    #         gamma=0.1)  # 学习率衰减的乘法因子

]

# runtime settings
checkpoint_config = dict(interval=10)
# work_dir = './work_dirs/sthv2_swin_base_patch244_window1677.py'
find_unused_parameters = True
auto_scale_lr = dict(enable=False, base_batch_size=128)


# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
