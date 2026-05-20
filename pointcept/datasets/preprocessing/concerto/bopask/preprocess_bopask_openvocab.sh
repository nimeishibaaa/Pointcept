#!/bin/bash

# ── 默认路径，按实际修改 ──────────────────────────────────────────
src_dir="/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train"
out_dir="/gpfs/work/mec/yiwenwang18/Pointcept/data/concerto/bopask_openvocab"
bop_original_dir="/gpfs/work/mec/yiwenwang18/datasets/bop_original/handal"
split_type="val"   # BopAsk 训练集对应 BOP 的 val split

while getopts "s:o:b:p:h" opt; do
  case $opt in
    s) src_dir=$OPTARG ;;
    o) out_dir=$OPTARG ;;
    b) bop_original_dir=$OPTARG ;;
    p) split_type=$OPTARG ;;
    h) echo "Usage: $0 [-s src_dir] [-o out_dir] [-b bop_original_dir] [-p split_type]"; exit 0 ;;
    *) echo "Usage: $0 [-s src_dir] [-o out_dir] [-b bop_original_dir] [-p split_type]"; exit 1 ;;
  esac
done

echo "================================================================"
echo "BopAsk Preprocessing Pipeline"
echo "  src_dir          : $src_dir"
echo "  out_dir          : $out_dir"
echo "  bop_original_dir : $bop_original_dir"
echo "  split_type       : $split_type"
echo "================================================================"

# Step 1: 点云生成 + labels.json
echo "[1/2] Generating point clouds and labels.json ..."
python pointcept/datasets/preprocessing/concerto/bopask/preprocess_bopask_openvocab.py \
    --src_dir         "$src_dir" \
    --out_dir         "$out_dir" \
    --bop_original_dir "$bop_original_dir" \
    --split_type      "$split_type"

if [ $? -ne 0 ]; then
    echo "ERROR: preprocess_bopask.py failed!"
    exit 1
fi

# Step 2: 场景级 train/val 划分
echo ""
echo "[2/2] Generating scene-level train/val splits ..."
python pointcept/datasets/preprocessing/concerto/bopask/splits_scenes.py \
    --dataset_root "$out_dir" \
    --val_ratio 0.1

echo "================================================================"
echo "Done. Data ready at: $out_dir"
echo "================================================================"