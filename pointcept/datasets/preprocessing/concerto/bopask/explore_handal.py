import json
import os
import shutil

src_dir = '/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train'
dst_dir = '/gpfs/work/mec/yiwenwang18/datasets/bopask-train/handal'

# Check first line of JSONL
jsonl_path = os.path.join(dst_dir, 'bopask-handal-train.jsonl')
if os.path.exists(jsonl_path):
    with open(jsonl_path, 'r') as f:
        first_line = json.loads(f.readline())
        print("Keys in annotation:", first_line.keys())
        # Print camera intrinsics if available
        if 'camera' in first_line:
            print("Camera info:", first_line['camera'])
        elif 'intrinsics' in first_line:
            print("Intrinsics:", first_line['intrinsics'])
        elif 'cam_K' in first_line:
            print("Cam_K:", first_line['cam_K'])
else:
    print(f"File not found: {jsonl_path}")

# Plan move
dirs_to_move = ['images', 'depth_maps', 'masks']
for d in dirs_to_move:
    src_d = os.path.join(src_dir, d)
    dst_d = os.path.join(dst_dir, d)
    if os.path.exists(src_d):
        print(f"Found {src_d}, moving to {dst_d}...")
        if not os.path.exists(dst_d):
            shutil.move(src_d, dst_d)
            print(f"Moved {d}")
        else:
            print(f"{dst_d} already exists. Skipping move.")
    else:
        print(f"Not found: {src_d}")
