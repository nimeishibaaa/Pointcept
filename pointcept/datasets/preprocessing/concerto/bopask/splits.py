import os
import json
import argparse
import glob

def get_splits_paths(dataset_path):
    # dataset_path is e.g. Pointcept/data/pointcept/handal
    im_path = os.path.join(dataset_path, "images")
    pc_path = dataset_path
    
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)
    
    # In HANDAL, scenes are directly under images/
    if not os.path.exists(im_path):
        print(f"Images path {im_path} does not exist. Run preprocess_handal.py first.")
        return
        
    scene_names = [f.name for f in os.scandir(im_path) if f.is_dir()]
    
    # We create a single 'train' split for now (or could separate based on scene IDs)
    split_dict = {}
    
    for scene_name in scene_names:
        # e.g. scene_000001
        cam_split_name_path = os.path.join(im_path, scene_name)
        cam_names = [f.name for f in os.scandir(cam_split_name_path) if f.is_dir()]
        
        for cam_name in cam_names:
            # e.g. camera_1
            im_split_name_path = os.path.join(cam_split_name_path, cam_name, "color")
            co_split_name_path = os.path.join(cam_split_name_path, cam_name, "correspondence")
            
            if not os.path.exists(im_split_name_path):
                continue
                
            png_files = [f for f in os.listdir(im_split_name_path) if f.endswith(".png")]
            # Sort by frame ID
            png_files = sorted(png_files, key=lambda x: int(x.split(".")[0]))
            
            png_file_paths = [
                os.path.join(im_split_name_path, f).replace(dataset_path, "data/handal")
                for f in png_files
            ]
            
            co_file_paths = [
                os.path.join(co_split_name_path, f.replace(".png", ".npy")).replace(dataset_path, "data/handal")
                for f in png_files
            ]
            
            # Group by 4 frames as in Concerto standard
            group_size = 4
            for i in range(0, len(png_file_paths), group_size):
                if i + group_size > len(png_file_paths):
                    continue # Skip incomplete groups
                    
                chunk_key = f"{scene_name}_{cam_name}_{i//group_size}"
                split_dict[chunk_key] = {
                    "pointclouds": os.path.join(pc_path, scene_name).replace(dataset_path, "data/handal"),
                    "images": png_file_paths[i : i + group_size],
                    "correspondences": co_file_paths[i : i + group_size]
                }
                
    with open(os.path.join(split_path, "train.json"), "w") as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"Successfully generated splits/train.json with {len(split_dict)} chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the preprocessed handal dataset (e.g. data/pointcept/handal)",
    )
    config = parser.parse_args()
    get_splits_paths(config.dataset_root)