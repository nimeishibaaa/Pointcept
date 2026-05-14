import os
import json
import argparse
import glob
import numpy as np

def generate_splits(dataset_path):
    # dataset_path is e.g. /gpfs/work/mec/yiwenwang18/Pointcept/data/pointcept/bopask
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)
    
    # Find all frame directories: dataset_path/scene_*/scene_*_frame_*
    frame_dirs = glob.glob(os.path.join(dataset_path, "scene_*", "scene_*_frame_*"))
    
    if not frame_dirs:
        print(f"No frame directories found in {dataset_path}. Run preprocess_bopask.py first.")
        return
        
    print(f"Found {len(frame_dirs)} valid preprocessed frames.")
    
    # Shuffle and split 90/10
    np.random.seed(42)
    np.random.shuffle(frame_dirs)
    split_idx = int(len(frame_dirs) * 0.9)
    train_frames = frame_dirs[:split_idx]
    val_frames = frame_dirs[split_idx:]
    
    def create_split_json(frames, out_file):
        split_dict = {}
        # Scene statistics tracker
        scene_counts = {}
        
        for i, frame_path in enumerate(frames):
            # Extract scene_id for statistics
            # frame_path looks like: .../scene_000001/scene_000001_frame_000001
            scene_id = os.path.basename(os.path.dirname(frame_path))
            scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
            
            # Convert to relative path from dataset_root
            rel_path = os.path.relpath(frame_path, dataset_path)
            
            # Use relative path as key and pointclouds value
            # Note: We must ensure it starts with "data/pointcept/bopask/" to match S3DIS logic 
            # if Pointcept expects absolute-like dataset paths, but usually relative is fine.
            # We'll use the raw relative path since HandalDataset will join it with data_root.
            split_dict[f"item_{i}"] = {
                "pointclouds": rel_path
            }
        
        with open(os.path.join(split_path, out_file), "w") as f:
            json.dump(split_dict, f, indent=4)
        
        print(f"\n[{out_file}] Total Samples: {len(frames)}")
        print(f"[{out_file}] Scene Distribution:")
        for sid, count in sorted(scene_counts.items()):
            print(f"  - {sid}: {count} frames")

    create_split_json(train_frames, "train.json")
    create_split_json(val_frames, "val.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the preprocessed bopask dataset",
    )
    config = parser.parse_args()
    generate_splits(config.dataset_root)