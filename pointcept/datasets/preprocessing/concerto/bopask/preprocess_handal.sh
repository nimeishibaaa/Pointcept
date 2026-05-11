#!/bin/bash

# Default paths
src_dir="/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train"
bop_val_dir="/gpfs/work/mec/yiwenwang18/datasets/bop_original/handal/val"
output_root="/gpfs/work/mec/yiwenwang18/Pointcept/data/pointcept/handal"
rgb_gap=5
voxel_size=0.005

while getopts "s:b:o:g:v:" opt; do
  case $opt in
    s) src_dir=$OPTARG ;;  
    b) bop_val_dir=$OPTARG ;;   
    o) output_root=$OPTARG ;;   
    g) rgb_gap=$OPTARG ;;  
    v) voxel_size=$OPTARG ;; 
    *) echo "Usage: $0 [-s <src_dir>] [-b <bop_val_dir>] [-o <output_root>] [-g <rgb_gap>] [-v <voxel_size>]"; exit 1 ;;
  esac
done

echo "==============================================="
echo "Starting HANDAL 2D-3D Preprocessing Pipeline"
echo "==============================================="
echo "Source Dir: $src_dir"
echo "BOP Val Dir: $bop_val_dir"
echo "Output Root: $output_root"
echo "RGB Gap: $rgb_gap"
echo "Voxel Size: $voxel_size"
echo "-----------------------------------------------"

# Step 0: Clean up previous generated data
echo "[0/3] Cleaning up previous data in output root..."
if [ -d "$output_root" ]; then
    rm -rf "$output_root/scene_"*
    rm -rf "$output_root/images"
    rm -rf "$output_root/splits"
    echo "Cleanup complete."
else
    echo "No previous data found. Skipping cleanup."
fi

# Step 1: Run the main 2D to 3D global unprojection and KDTree correspondence matching
echo "[1/3] Running global unprojection and correspondence matching..."
export OMP_NUM_THREADS=2  # Limit OpenMP threads further
python pointcept/datasets/preprocessing/concerto/bopask/preprocess_handal.py \
    --src_dir "$src_dir" \
    --bop_val_dir "$bop_val_dir" \
    --output_root "$output_root" \
    --rgb_gap "$rgb_gap" \
    --voxel_size "$voxel_size"

if [ $? -ne 0 ]; then
    echo "Error: preprocess_handal.py failed."
    exit 1
fi

# Step 2: Generate the JSON split file for Concerto Dataloader
echo "[2/3] Generating split JSON files..."
python pointcept/datasets/preprocessing/concerto/bopask/splits.py \
    --dataset_root "$output_root"

if [ $? -ne 0 ]; then
    echo "Error: splits.py failed."
    exit 1
fi

# Step 3: Generate PLY visualizations
echo "[3/3] Generating PLY visualizations for CloudCompare..."
python pointcept/datasets/preprocessing/concerto/bopask/visualize_npy.py \
    --dataset_root "$output_root"

if [ $? -ne 0 ]; then
    echo "Warning: visualize_npy.py failed, but preprocessing is complete."
fi

echo "==============================================="
echo "Preprocessing Complete!"
echo "PLY visualizations are saved in $output_root"
echo "You can now use 'data/handal' in your Pointcept configs."
echo "==============================================="
