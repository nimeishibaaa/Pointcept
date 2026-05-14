#!/bin/bash

# Configuration
src_dir="/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train"
out_dir="/gpfs/work/mec/yiwenwang18/Pointcept/data/concerto/bopask"
bop_original_dir="/gpfs/work/mec/yiwenwang18/datasets/bop_original/handal"
split_type="val"  # Handal train scenes in BOPAsk actually come from BOP's val set
num_workers=8

# Usage print
while getopts "s:o:b:p:n:h" opt; do
  case $opt in
    s) src_dir=$OPTARG ;;  
    o) out_dir=$OPTARG ;;   
    b) bop_original_dir=$OPTARG ;;
    p) split_type=$OPTARG ;;
    n) num_workers=$OPTARG ;;   
    h) echo "Usage: $0 [-s src_dir] [-o out_dir] [-b bop_original_dir] [-p split_type] [-n num_workers]"; exit 0 ;;
    *) echo "Usage: $0 [-s src_dir] [-o out_dir] [-b bop_original_dir] [-p split_type] [-n num_workers]"; exit 1 ;;
  esac
done

echo "================================================================"
echo "Starting BOPAsk Preprocessing Pipeline"
echo "Source Dir: $src_dir"
echo "Output Dir: $out_dir"
echo "BOP Original Dir: $bop_original_dir"
echo "Split Type: $split_type"
echo "================================================================"

# Step 1: Run the main python preprocessing script
echo "[1/2] Running Point Cloud Generation and Filtering..."
python pointcept/datasets/preprocessing/concerto/bopask/preprocess_bopask.py \
    --src_dir "$src_dir" \
    --out_dir "$out_dir" \
    --bop_original_dir "$bop_original_dir" \
    --split_type "$split_type"

if [ $? -ne 0 ]; then
    echo "Error: preprocess_bopask.py failed!"
    exit 1
fi

# Step 2: Generate the dataset splits and print statistics
echo ""
echo "[2/2] Generating Train/Val Splits and Statistics..."
python pointcept/datasets/preprocessing/concerto/bopask/splits.py \
    --dataset_root "$out_dir"

echo "================================================================"
echo "Preprocessing Pipeline Completed Successfully!"
echo "Data is ready at: $out_dir"