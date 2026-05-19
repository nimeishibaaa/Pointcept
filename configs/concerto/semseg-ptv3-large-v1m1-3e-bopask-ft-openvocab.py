_base_ = ["../_base_/default_runtime.py"]

# disable wandb
enable_wandb = False

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 8
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True

# trainer
train = dict(
    type="DefaultTrainer",
)

# Open-Vocabulary PPT Model Configuration
model = dict(
    type="OpenVocabPPT",
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
        freeze_encoder=True, # You can keep the encoder frozen!
    ),
    # IMPORTANT: We replaced CrossEntropyLoss with BinaryFocalLoss for open-vocab matching
    criteria=[
        dict(type="BinaryFocalLoss", gamma=2.0, alpha=0.5, logits=True, loss_weight=1.0),
    ],
    freeze_backbone=False,
    backbone_out_channels=64,
    template="[x]",
    clip_model="ViT-B/16",
)

# scheduler settings
epoch = 100
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

# NOTE ON DATASET MODIFICATION:
# To use this Open-Vocab framework, your HandalDataset (or its transform pipeline) MUST be updated:
# 1. Provide `data_dict["texts"]`: a list of strings (e.g., ["background", "red spatula", "cookie jar"]).
# 2. Provide `data_dict["segment"]`: a binary mask of shape [N_points, Num_texts] (0s and 1s).
# 3. DO NOT map background to -1 anymore. Include background as a class so the model learns it.
data = dict(
    num_classes=0, # Not used in open-vocab, but kept for compatibility
    ignore_index=-1,
    names=[], # Not used, texts are dynamic
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                split="train",
                data_root=data_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    # dict(type="MapLabel", mapping_dict=mapping_dict), # REMOVED closed-set mapping
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
                    # Needs a custom transform here to inject `texts` and convert `segment` to binary mask
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "texts"), # ADDED texts
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
            # dict(type="MapLabel", mapping_dict=mapping_dict),
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "texts"),
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
                    keys=("coord", "grid_coord", "index", "texts"),
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
    dict(type="OpenVocabEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
