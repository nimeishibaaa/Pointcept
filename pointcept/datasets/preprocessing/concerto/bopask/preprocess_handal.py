"""
Preprocessing Script for HANDAL (BOP format) to Concerto S3DIS format

This script converts BOP format datasets (like HANDAL) into a globally consistent
3D point cloud format with 2D-3D correspondences for Concerto pretraining/finetuning.
"""

import os
import argparse
import glob
import numpy as np
import json
from PIL import Image
from scipy.spatial import cKDTree
from pathlib import Path
import shutil

try:
    import open3d as o3d
except ImportError:
    import warnings
    warnings.warn("Please install open3d for parsing normal and voxel downsampling")

from tqdm import tqdm


def process_scene(scene_id, images_dir, depth_dir, masks_dir, camera_json_path, output_root, rgb_gap=5, voxel_size=0.005):
    print(f"Parsing scene: {scene_id}")
    
    with open(camera_json_path, 'r') as f:
        cam_data = json.load(f)
        
    # Find all rgb frames
    rgb_files = sorted(glob.glob(os.path.join(images_dir, f"scene_{scene_id}_frame_*.png")))
    rgb_files = rgb_files[::rgb_gap]
    
    if not rgb_files:
        print(f"No rgb files found for scene {scene_id}")
        return
        
    scene_output_dir = Path(output_root) / f"scene_{scene_id}"
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    cam_output_dir = Path(output_root) / "images" / f"scene_{scene_id}" / "camera_1"
    for sub in ['color', 'depth', 'intrinsic', 'pose', 'correspondence']:
        (cam_output_dir / sub).mkdir(parents=True, exist_ok=True)
        
    global_coords = []
    global_colors = []
    global_segments = []
    global_instances = []
    
    frame_data_list = []
    
    for rgb_path in tqdm(rgb_files, desc="Unprojecting frames"):
        basename = os.path.basename(rgb_path)
        frame_id_str = basename.split('_')[3].replace('.png', '')
        frame_id_int = str(int(frame_id_str)) # removing leading zeros for json key
        
        if frame_id_int not in cam_data:
            continue
            
        frame_cam = cam_data[frame_id_int]
        K = np.array(frame_cam['cam_K']).reshape(3, 3)
        R_w2c = np.array(frame_cam['cam_R_w2c']).reshape(3, 3)
        t_w2c = np.array(frame_cam['cam_t_w2c']).reshape(3) / 1000.0 # Convert mm to meters
        
        # Build T_c2w (Camera to World)
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c
        T_c2w = np.linalg.inv(T_w2c)
        
        depth_scale = frame_cam.get('depth_scale', 1.0)
        
        depth_path = os.path.join(depth_dir, basename.replace('.png', '_depth.png'))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(depth_dir, basename)
        if not os.path.exists(depth_path):
            continue
            
        # Masks
        mask_pattern = os.path.join(masks_dir, f"scene_{scene_id}_frame_{frame_id_str}_*_mask.png")
        mask_files = glob.glob(mask_pattern)
        
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))
        depth_img = np.array(Image.open(depth_path))
        H, W = depth_img.shape
        
        segment_img = np.ones((H, W), dtype=np.int32) * -1
        instance_img = np.ones((H, W), dtype=np.int32) * -1
        
        for mf in mask_files:
            parts = os.path.basename(mf).split('_')
            try:
                idx = parts.index('mask.png')
                obj_id = int(parts[idx-1]) # target_{obj_id}
                mask_img = np.array(Image.open(mf))
                valid_mask = mask_img > 0
                segment_img[valid_mask] = obj_id
                instance_img[valid_mask] = obj_id # In BOP, obj_id differentiates instances/classes
            except:
                continue
                
        # Unproject to Camera Space
        depth_in_m = (depth_img * depth_scale) / 1000.0 # Convert mm to meters
        
        # Dynamic depth cutoff based on object masks to handle tilted cameras
        has_objects = (segment_img > 0)
        if has_objects.any():
            # Find the furthest object point and add 1.0m margin for the table/context
            max_obj_depth = depth_in_m[has_objects].max()
            depth_limit = max_obj_depth + 1.0
        else:
            # Fallback if no objects in view
            depth_limit = 3.0
            
        valid = (depth_img > 0) & (depth_in_m < depth_limit)
        z = depth_in_m[valid]
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u_valid = u[valid]
        v_valid = v[valid]
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy
        
        pts_cam = np.stack((x, y, z), axis=1) # (N, 3)
        
        # Transform to World Space
        pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_world = (T_c2w @ pts_cam_homo.T).T[:, :3]
        
        # Ensure we filter out any NaNs or Infs that could cause segfaults later
        valid_pts = ~np.isnan(pts_world).any(axis=1) & ~np.isinf(pts_world).any(axis=1)
        if not valid_pts.all():
            pts_world = pts_world[valid_pts]
            colors = rgb_img[valid][valid_pts]
            segments = segment_img[valid][valid_pts]
            instances = instance_img[valid][valid_pts]
            u_valid = u_valid[valid_pts]
            v_valid = v_valid[valid_pts]
        else:
            colors = rgb_img[valid]
            segments = segment_img[valid]
            instances = instance_img[valid]
        
        global_coords.append(pts_world)
        global_colors.append(colors)
        global_segments.append(segments)
        global_instances.append(instances)
        
        frame_data_list.append({
            'id': frame_id_str,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'K': K,
            'T_c2w': T_c2w,
            'u': u_valid,
            'v': v_valid,
            'pts_world': pts_world
        })
        
    if not global_coords:
        return
        
    global_coords = np.vstack(global_coords).astype(np.float32)
    global_colors = np.vstack(global_colors).astype(np.uint8)
    global_segments = np.concatenate(global_segments).reshape(-1, 1).astype(np.int16)
    global_instances = np.concatenate(global_instances).reshape(-1, 1).astype(np.int16)
    
    # Voxel Downsample
    print(f"Voxel downsampling from {len(global_coords)} points...")
    voxel_indices = np.floor(global_coords / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    # If the point cloud is still absurdly large after voxelization, we need to enforce a hard cap
    # to prevent Open3D from segfaulting during normal estimation.
    MAX_POINTS_CAP = 10_000_000 # 10 million points max per scene
    if len(unique_indices) > MAX_POINTS_CAP:
        print(f"WARNING: Point cloud still too large ({len(unique_indices)}). Randomly subsampling to {MAX_POINTS_CAP}...")
        
        # Smart Subsampling: Prioritize points that belong to objects
        is_object = global_segments[unique_indices].squeeze() > 0
        obj_indices = np.where(is_object)[0]
        bg_indices = np.where(~is_object)[0]
        
        np.random.seed(42)
        if len(obj_indices) >= MAX_POINTS_CAP:
            # Extremely rare: even the objects themselves have >10M points
            chosen_indices = np.random.choice(obj_indices, MAX_POINTS_CAP, replace=False)
        else:
            # Normal case: keep ALL object points, and randomly sample the remaining quota from background
            remaining_cap = MAX_POINTS_CAP - len(obj_indices)
            chosen_bg = np.random.choice(bg_indices, remaining_cap, replace=False)
            chosen_indices = np.concatenate([obj_indices, chosen_bg])
            np.random.shuffle(chosen_indices) # Shuffle to mix them
            
        unique_indices = unique_indices[chosen_indices]

    down_coords = global_coords[unique_indices].astype(np.float64) # Force float64 for Open3D
    down_colors = global_colors[unique_indices]
    down_segments = global_segments[unique_indices]
    down_instances = global_instances[unique_indices]
    
    print(f"Points remaining after downsampling: {len(down_coords)}")
    
    # Estimate Normals
    print("Estimating normals...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(down_coords)
    # Reduce max_nn to prevent memory exhaustion on supercomputer nodes
    # For large datasets, use a smaller max_nn and limit thread count further if needed.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=10))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.])) # Approximation for normal orientation
    down_normals = np.asarray(pcd.normals).astype(np.float32)
    
    # Save Global Point Cloud
    np.save(scene_output_dir / "coord.npy", down_coords)
    np.save(scene_output_dir / "color.npy", down_colors)
    np.save(scene_output_dir / "segment.npy", down_segments)
    np.save(scene_output_dir / "instance.npy", down_instances)
    np.save(scene_output_dir / "normal.npy", down_normals)
    
    # Correspondences & Multi-view Data
    print("Building KDTree for correspondences...")
    tree = cKDTree(down_coords)
    
    for frame in tqdm(frame_data_list, desc="Saving multi-view data"):
        fid = frame['id']
        # Copy images
        shutil.copy2(frame['rgb_path'], cam_output_dir / "color" / f"{fid}.png")
        shutil.copy2(frame['depth_path'], cam_output_dir / "depth" / f"{fid}.png")
        
        # Save K, T
        np.save(cam_output_dir / "intrinsic" / f"{fid}.npy", frame['K'])
        np.save(cam_output_dir / "pose" / f"{fid}.npy", frame['T_c2w'])
        
        # Calculate Correspondences
        dists, idxs = tree.query(frame['pts_world'], k=1)
        valid_corr = dists < (voxel_size * 2) # Threshold based on voxel size
        
        u_valid = frame['u'][valid_corr]
        v_valid = frame['v'][valid_corr]
        idx_valid = idxs[valid_corr]
        
        if len(u_valid) > 0:
            correspondences = np.column_stack((u_valid, v_valid, idx_valid)).astype(np.float32)
        else:
            correspondences = -np.ones((1, 3), dtype=np.float32)

        np.save(cam_output_dir / "correspondence" / f"{fid}.npy", correspondences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess BOP format dataset to Pointcept MultiView format.")
    parser.add_argument("--src_dir", required=True, help="Directory containing images/, depth_maps/, masks/")
    parser.add_argument("--bop_val_dir", required=True, help="BOP original dir containing scene_camera.json (e.g. handal/val)")
    parser.add_argument("--output_root", required=True, help="Output directory for Concerto format")
    parser.add_argument("--rgb_gap", type=int, default=5, help="Frame sampling gap")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for downsampling (meters)")
    
    args = parser.parse_args()
    
    images_dir = os.path.join(args.src_dir, 'images')
    depth_dir = os.path.join(args.src_dir, 'depth_maps')
    masks_dir = os.path.join(args.src_dir, 'masks')
    
    # Find unique scenes
    all_imgs = glob.glob(os.path.join(images_dir, "*.png"))
    scene_ids = sorted(list(set([os.path.basename(f).split('_')[1] for f in all_imgs])))
    
    print(f"Found {len(scene_ids)} scenes to process.")
    
    for scene_id in scene_ids:
        camera_json = os.path.join(args.bop_val_dir, scene_id, 'scene_camera.json')
        if not os.path.exists(camera_json):
            print(f"Warning: {camera_json} not found. Skipping scene {scene_id}.")
            continue
            
        process_scene(scene_id, images_dir, depth_dir, masks_dir, camera_json, args.output_root, args.rgb_gap, args.voxel_size)
