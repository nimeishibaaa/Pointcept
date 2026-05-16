_base_ = ["../_base_/default_runtime.py"]

# disable wandb
enable_wandb = False

# 10 macro categories based on the 40 items
CLASS_LABELS_10 = [
    "hammer",             # 1-9
    "slotted spoon",      # 10
    "ladle",              # 11-12
    "spaghetti spoon",    # 13
    "measuring spoon",    # 14-19
    "power drill",        # 20-25
    "silicone spatula",   # 26-27
    "turner spatula",     # 28-29
    "strainer",           # 30-34
    "whisk"               # 35-40
]

# Generate mapping_dict: 
# 0 (obstacle) -> -1
# 41 (unknown) -> -1
# 1-40 -> 0-9 (mapped to their macro category)
mapping_dict = {0: -1, 41: -1}
for i in range(1, 10): mapping_dict[i] = 0   # hammer (1-9)
mapping_dict[10] = 1                         # slotted spoon (10)
for i in range(11, 13): mapping_dict[i] = 2  # ladle (11-12)
mapping_dict[13] = 3                         # spaghetti spoon (13)
for i in range(14, 20): mapping_dict[i] = 4  # measuring spoon (14-19)
for i in range(20, 26): mapping_dict[i] = 5  # electric drill (20-25)
for i in range(26, 28): mapping_dict[i] = 6  # silicone spatula (26-27)
for i in range(28, 30): mapping_dict[i] = 7  # turner spatula (28-29)
for i in range(30, 35): mapping_dict[i] = 8  # strainer (30-34)
for i in range(35, 41): mapping_dict[i] = 9  # whisk (35-40)

# misc custom setting
batch_size = 4  # bs: total bs in all gpus (reduced to 4 to prevent OOM)
num_worker = 8  # total workers
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True

# trainer
train = dict(
    type="DefaultTrainer",  # Add explicitly to prevent default bugs
)

# model settings
model = dict(
    type="DefaultLORASegmentorV2",
    num_classes=10,  # Changed to 10 macro classes
    backbone_out_channels=1728,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    keywords="module.student.backbone",
    replacements="module.backbone",
    backbone_path="exp/concerto/pretrained_model/pretrain-concerto-v1m1-2-large-video.pth",  # Fixed path
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,
        traceable=True,
        mask_token=False,
        freeze_encoder=True,
    ),
    criteria=[
        # Loss weight reset to 1.0 since background is removed
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

# scheduler settings
epoch = 100  # Updated from 3000 to match PPT runs
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "HandalDataset"
data_root = "data/concerto/bopask"

data = dict(
    num_classes=10,  # Changed to 10
    ignore_index=-1,
    names=CLASS_LABELS_10,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="MapLabel", mapping_dict=mapping_dict),  # Map to 10 classes & ignore bg
            dict(type="RandomDropout", dropout_ratio=0.1, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomJitter", sigma=0.0005, clip=0.0015),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
        loop=1,  # Added loop=1 to match dataset real size
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="MapLabel", mapping_dict=mapping_dict),  # Map to 10 classes
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),  # Added to retain origin
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="MapLabel", mapping_dict=mapping_dict),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

# hook
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
