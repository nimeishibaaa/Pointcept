import os
import json
import glob
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_point_cloud_from_rgbd(rgb_img, depth_img, masks_dict, fx, fy, cx, cy, depth_scale=1000.0):
    # rgb_img: (H, W, 3)
    # depth_img: (H, W)
    # masks_dict: dict {class_id: mask_img (H, W)}
    H, W = depth_img.shape
    
    # Generate pixel coordinates
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Valid depth mask
    valid = depth_img > 0
    
    # Project to 3D
    z = depth_img[valid] / depth_scale
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    coord = np.stack((x, y, z), axis=1) # (N, 3)
    color = rgb_img[valid] # (N, 3)
    
    # Compute segment labels
    segment = np.ones((H, W), dtype=np.int32) * -1 # ignore_index
    for class_id, mask in masks_dict.items():
        segment[mask > 0] = class_id
    segment = segment[valid] # (N,)
    
    # normals (zeros for now, Pointcept can compute or ignore if not used)
    normal = np.zeros_like(coord)
    
    return coord, color, segment, normal

def main():
    src_dir = '/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train'
    dst_dir = '/gpfs/work/mec/yiwenwang18/datasets/bopask-train/handal'
    
    # 1. Reorganize files using symbolic links
    print("Checking directories...")
    for sub in ['images', 'depth_maps', 'masks']:
        os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)
        src_sub = os.path.join(src_dir, sub)
        if os.path.exists(src_sub):
            files = glob.glob(os.path.join(src_sub, '*.*'))
            print(f"Symlinking {len(files)} files from {src_sub} to {os.path.join(dst_dir, sub)}...")
            for f in tqdm(files, desc=f"Symlinking {sub}"):
                dst_file = os.path.join(dst_dir, sub, os.path.basename(f))
                if not os.path.exists(dst_file):
                    try:
                        os.symlink(f, dst_file)
                    except Exception as e:
                        pass

    # 2. Process to Pointcept format
    # Using relative path for generated Pointcept/concerto data
    pointcept_out_dir = '../../../../../data/pointcept/handal'
    # Resolving it relative to the script's directory so it works regardless of where it is run from
    pointcept_out_dir = os.path.join(os.path.dirname(__file__), pointcept_out_dir)
    
    train_out = os.path.join(pointcept_out_dir, 'train')
    val_out = os.path.join(pointcept_out_dir, 'val')
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)
    
    images = glob.glob(os.path.join(dst_dir, 'images', '*.png'))
    print(f"Found {len(images)} images to process.")
    
    # simple split 90/10
    np.random.seed(42)
    np.random.shuffle(images)
    split_idx = int(len(images) * 0.9)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Intrinsics (approximate fallback)
    fallback_fx, fallback_fy = 1000.0, 1000.0
    fallback_cx, fallback_cy = 1920 / 2.0, 1440 / 2.0  # approximate image center
    
    for split, img_list in [('train', train_images), ('val', val_images)]:
        out_split_dir = train_out if split == 'train' else val_out
        for img_path in tqdm(img_list, desc=f"Processing {split}"):
            basename = os.path.basename(img_path)
            # scene_000001_frame_000001.png or similar
            name_stem = basename.replace('.png', '')
            
            # Parse scene ID to find scene_camera.json if available
            scene_id = name_stem.split('_')[1] # e.g. "000001"
            # In BOP, the real camera.json for validation set is usually in bop_original/handal/val/xxxxxx/scene_camera.json
            bop_val_dir = '/gpfs/work/mec/yiwenwang18/datasets/bop_original/handal/val'
            camera_json_path = os.path.join(bop_val_dir, scene_id, 'scene_camera.json')
            fx, fy = fallback_fx, fallback_fy
            cx, cy = fallback_cx, fallback_cy
            
            if os.path.exists(camera_json_path):
                try:
                    with open(camera_json_path, 'r') as f:
                        cam_data = json.load(f)
                        # Extract from BOP format: cam_data[frame_id]['cam_K']
                        frame_id = str(int(name_stem.split('_')[3]))
                        if frame_id in cam_data:
                            K = cam_data[frame_id]['cam_K']
                            fx, fy = K[0], K[4]
                            cx, cy = K[2], K[5]
                except Exception:
                    pass
            
            # Find corresponding depth
            # depth might be named scene_..._depth.png or scene_...png
            depth_path = os.path.join(dst_dir, 'depth_maps', basename.replace('.png', '_depth.png'))
            if not os.path.exists(depth_path):
                depth_path = os.path.join(dst_dir, 'depth_maps', basename)
            if not os.path.exists(depth_path):
                continue
                
            # Find masks
            # masks format: scene_000008_frame_000980_target_10_mask.png
            mask_pattern = os.path.join(dst_dir, 'masks', f"{name_stem}*_mask.png")
            mask_files = glob.glob(mask_pattern)
            
            masks_dict = {}
            for mf in mask_files:
                # extract class ID from filename, e.g. target_10_mask.png -> 10
                parts = os.path.basename(mf).split('_')
                try:
                    # heuristic to find the number before 'mask'
                    idx = parts.index('mask.png')
                    class_id = int(parts[idx-1])
                    mask_img = np.array(Image.open(mf))
                    masks_dict[class_id] = mask_img
                except:
                    continue
            
            rgb_img = np.array(Image.open(img_path).convert('RGB'))
            depth_img = np.array(Image.open(depth_path))
            
            coord, color, segment, normal = get_point_cloud_from_rgbd(
                rgb_img, depth_img, masks_dict, fx, fy, cx, cy
            )
            
            if len(coord) == 0:
                continue
                
            # Save to Pointcept format
            sample_dir = os.path.join(out_split_dir, name_stem)
            os.makedirs(sample_dir, exist_ok=True)
            
            np.save(os.path.join(sample_dir, 'coord.npy'), coord.astype(np.float32))
            np.save(os.path.join(sample_dir, 'color.npy'), color.astype(np.float32))
            np.save(os.path.join(sample_dir, 'segment.npy'), segment.astype(np.int32))
            np.save(os.path.join(sample_dir, 'normal.npy'), normal.astype(np.float32))

if __name__ == '__main__':
    main()
