_base_ = ["../_base_/default_runtime.py"]

# disable wandb
enable_wandb = False

CLASS_LABELS_BOPASK = [
    "obstacle",
    # 1-9: Hammers
    "hammer", "hammer", "hammer", "hammer", "hammer", "hammer", "hammer", "hammer", "hammer",
    # 10-14: Spatulas / Spoons
    "spatula", "spatula", "spatula", "spatula", "spatula",
    # 15-19: Measuring spoons
    "measuring spoon", "measuring spoon", "measuring spoon", "measuring spoon", "measuring spoon",
    # 20-26: Power drills
    "power drill", "power drill", "power drill", "power drill", "power drill", "power drill", "power drill",
    # 27-30: Ladles
    "ladle", "ladle", "ladle", "ladle",
    # 31-34: Strainers
    "strainer", "strainer", "strainer", "strainer",
    # 35-40: Whisks
    "whisk", "whisk", "whisk", "whisk", "whisk", "whisk"
]
# BOPAsk/Handal object classes (example placeholders, update with real names if available)
# 手动构建的 HANDAL 类别文本 (根据 BOP 2024 论文图示)
# 索引 0 作为背景占位，索引 1-40 对应 obj_000001 到 obj_000040，索引41是全部未知物体


# misc custom setting
batch_size = 4  # bs: total bs in all gpus (reduced from 8 to avoid OOM on 2 GPUs during gradient accumulation or uneven splits)
num_worker = 8
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="PPT-v1m3",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        traceable=True,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
    backbone_out_channels=64,
    conditions=("BOPAsk",),
    template="[x]",
    clip_model="ViT-B/16",
    class_names=[CLASS_LABELS_BOPASK],  # 全部参与loss计算 (41个类: 0=obstacle, 1-40=objects)
    backbone_mode=False,
)

# scheduler settings
epoch = 3000
eval_epoch = 300
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
    num_classes=41,
    ignore_index=-1,
    names=CLASS_LABELS_BOPASK,
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                split="train",
                data_root=data_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(type="MapLabel", mapping_dict={41: 0}),
            dict(type="RandomDropout", dropout_ratio=0.1, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            # dict(type="RandomFlip", p=0.5), # 禁用翻转以保护相机射线和物体手性
            dict(type="RandomJitter", sigma=0.0005, clip=0.0015),
            # dict(type="ElasticDistortion", distortion_params=[[0.05, 0.1], [0.2, 0.4]]),
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
                    # KEEP NORMAL: Camera Ray Encoding is preserved!
                    dict(type="Update", keys_dict={"condition": "BOPAsk"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=1,
            ),
        ]
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="MapLabel", mapping_dict={41: 0}),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # KEEP NORMAL
            dict(type="Update", keys_dict={"condition": "BOPAsk"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "condition", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            # KEEP NORMAL
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
                dict(type="Update", keys_dict={"condition": "BOPAsk"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
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
