import os
import json
import shutil
import glob

def prepare_data():
    src_dir = '/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train'
    dst_dir = '/gpfs/work/mec/yiwenwang18/datasets/bopask-train/handal'
    jsonl_path = os.path.join(dst_dir, 'bopask-handal-train.jsonl')
    
    print(f"Creating directories in {dst_dir}...")
    for sub in ['images', 'depth_maps', 'masks']:
        os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)
    
    print("Moving files from src to dst...")
    # Move images
    for sub in ['images', 'depth_maps', 'masks']:
        src_sub = os.path.join(src_dir, sub)
        dst_sub = os.path.join(dst_dir, sub)
        if os.path.exists(src_sub):
            files = glob.glob(os.path.join(src_sub, '*.*'))
            print(f"Found {len(files)} files in {src_sub}")
            for f in files:
                shutil.move(f, dst_sub)
                
    # Verify with JSONL
    print(f"Verifying against {jsonl_path}...")
    missing_images = 0
    total_lines = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                total_lines += 1
                data = json.loads(line)
                for img in data.get('images', []):
                    # img path in json is like "images/scene_000001_frame_000001.png"
                    img_path = os.path.join(dst_dir, img)
                    if not os.path.exists(img_path):
                        missing_images += 1
                if total_lines >= 1000: # just check first 1000 for speed
                    break
    print(f"Checked {total_lines} lines. Missing images: {missing_images}")

if __name__ == "__main__":
    prepare_data()
