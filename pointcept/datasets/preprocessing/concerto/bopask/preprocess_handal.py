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
    
    # ---------------------------------------------------------
    # BOP-Ask Paper Section 3.2.1: World frame construction
    # The original BOP world frame (from cam_R_w2c) is often arbitrary and the table is tilted.
    # We will estimate a Scene-Level Alignment Matrix (R_align) by fitting a plane to the 
    # objects in the first valid frame, aligning the table normal to the Z-axis [0,0,1].
    # ---------------------------------------------------------
    R_align = np.eye(3)
    alignment_computed = False
    
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
        
        has_objects = (segment_img > 0)
        if not has_objects.any():
            continue
            
        # 3D LOCAL CONTEXT BOUNDING BOX (ORIENTED BY WORLD COORDINATES): 
        # BOP datasets define the World Coordinate System such that the Z-axis is perpendicular to the table.
        # By transforming points to World Space first, our Axis-Aligned Bounding Box (AABB) in World Space
        # acts as an Oriented Bounding Box (OBB) in Camera Space. This perfectly handles tilted cameras.
        
        # 1. Unproject ONLY the object points first to find their 3D bounds
        v_obj, u_obj = np.where(has_objects & (depth_img > 0))
        if len(v_obj) == 0:
            continue
        z_obj = depth_in_m[v_obj, u_obj]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x_obj = (u_obj - cx) * z_obj / fx
        y_obj = (v_obj - cy) * z_obj / fy
        
        # Transform object points to World Space
        pts_obj_cam = np.stack((x_obj, y_obj, z_obj), axis=1)
        pts_obj_cam_homo = np.hstack((pts_obj_cam, np.ones((pts_obj_cam.shape[0], 1))))
        pts_obj_world = (T_c2w @ pts_obj_cam_homo.T).T[:, :3]
        
        # --- Compute Scene-Level Alignment (Once per scene) ---
        if not alignment_computed:
            try:
                # To prevent Open3D Segmentation Faults on the cluster, we use pure numpy SVD to fit the plane
                # since the object points are mostly spread across the table surface.
                pts_for_fit = pts_obj_world[::5] # Subsample for speed
                
                # 1. Calculate centroid and center the points
                centroid = np.mean(pts_for_fit, axis=0)
                centered_pts = pts_for_fit - centroid
                
                # 2. Compute SVD
                u, s, vh = np.linalg.svd(centered_pts, full_matrices=False)
                
                # The normal is the last row of Vh (corresponding to the smallest singular value)
                n_p = vh[2, :]
                
                # Ensure normal points "up" (Z > 0) or towards the camera
                if n_p[2] < 0:
                    n_p = -n_p
                    
                v_z = np.array([0.0, 0.0, 1.0])
                v = np.cross(n_p, v_z)
                c = np.dot(n_p, v_z)
                if np.linalg.norm(v) < 1e-6:
                    R_align = np.eye(3)
                else:
                    s_norm = np.linalg.norm(v)
                    kmat = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    R_align = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s_norm ** 2))
                print(f"\nScene Alignment Computed (Numpy SVD). Table normal in World Space: {n_p}")
            except Exception as e:
                print(f"\nWarning: SVD plane fitting failed ({e}), using default World Space.")
                R_align = np.eye(3)
            alignment_computed = True
        
        # Apply the alignment rotation so the table is flat (Z is up)
        pts_obj_aligned = (R_align @ pts_obj_world.T).T
        
        # Object bounding box in Aligned Space (Z is aligned with table normal)
        min_x, max_x = pts_obj_aligned[:, 0].min(), pts_obj_aligned[:, 0].max()
        min_y, max_y = pts_obj_aligned[:, 1].min(), pts_obj_aligned[:, 1].max()
        min_z, max_z = pts_obj_aligned[:, 2].min(), pts_obj_aligned[:, 2].max()
        
        # 2. Expand bounding box to include the supporting table and local context
        # (e.g., 0.5m padding in X/Y, 0.5m above, 0.2m below to capture table surface)
        margin_x, margin_y, margin_z_below, margin_z_above = 0.5, 0.5, 0.2, 0.5
        
        # 3. Unproject ALL valid depth points
        valid_depth = (depth_img > 0)
        v_all, u_all = np.where(valid_depth)
        z_all = depth_in_m[v_all, u_all]
        x_all = (u_all - cx) * z_all / fx
        y_all = (v_all - cy) * z_all / fy
        
        pts_all_cam = np.stack((x_all, y_all, z_all), axis=1)
        pts_all_cam_homo = np.hstack((pts_all_cam, np.ones((pts_all_cam.shape[0], 1))))
        pts_all_world = (T_c2w @ pts_all_cam_homo.T).T[:, :3]
        pts_all_aligned = (R_align @ pts_all_world.T).T
        
        # 4. Filter points that fall within the expanded Aligned Space Bounding Box
        in_box = (
            (pts_all_aligned[:, 0] >= min_x - margin_x) & (pts_all_aligned[:, 0] <= max_x + margin_x) &
            (pts_all_aligned[:, 1] >= min_y - margin_y) & (pts_all_aligned[:, 1] <= max_y + margin_y) &
            (pts_all_aligned[:, 2] >= min_z - margin_z_below) & (pts_all_aligned[:, 2] <= max_z + margin_z_above)
        )
        
        # Reconstruct the final valid arrays
        z = z_all[in_box]
        x = x_all[in_box]
        y = y_all[in_box]
        u_valid = u_all[in_box]
        v_valid = v_all[in_box]
        
        pts_cam = np.stack((x, y, z), axis=1) # (N, 3)
        
        # Transform to Original World Space, then apply Scene Alignment to make table flat (Z is up)
        pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_world_orig = (T_c2w @ pts_cam_homo.T).T[:, :3]
        pts_world = (R_align @ pts_world_orig.T).T
        
        # Update T_c2w to reflect the new aligned world space
        T_c2w_aligned = np.eye(4)
        T_c2w_aligned[:3, :3] = R_align @ T_c2w[:3, :3]
        T_c2w_aligned[:3, 3] = R_align @ T_c2w[:3, 3]
        
        # We need to map the in_box flat indices back to the 2D image 
        # to extract colors and segments properly
        valid_flat_mask = np.zeros_like(valid_depth, dtype=bool)
        valid_flat_mask[v_valid, u_valid] = True
        
        # Ensure we filter out any NaNs or Infs that could cause segfaults later
        valid_pts = ~np.isnan(pts_world).any(axis=1) & ~np.isinf(pts_world).any(axis=1)
        if not valid_pts.all():
            pts_world = pts_world[valid_pts]
            colors = rgb_img[valid_flat_mask][valid_pts]
            segments = segment_img[valid_flat_mask][valid_pts]
            instances = instance_img[valid_flat_mask][valid_pts]
            u_valid = u_valid[valid_pts]
            v_valid = v_valid[valid_pts]
        else:
            colors = rgb_img[valid_flat_mask]
            segments = segment_img[valid_flat_mask]
            instances = instance_img[valid_flat_mask]
        
        global_coords.append(pts_world)
        global_colors.append(colors)
        global_segments.append(segments)
        global_instances.append(instances)
        
        frame_data_list.append({
            'id': frame_id_str,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'K': K,
            'T_c2w': T_c2w_aligned,
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
    
    # Since we strictly filter out all background points, the total point count should be extremely small
    # (usually < 500k points per scene). The MAX_POINTS_CAP logic is kept just as a nuclear failsafe.
    MAX_POINTS_CAP = 3_000_000 # 3 million points
    if len(unique_indices) > MAX_POINTS_CAP:
        print(f"WARNING: Point cloud still too large ({len(unique_indices)}). Randomly subsampling to {MAX_POINTS_CAP}...")
        
        # Since all remaining points are object points, we just do a uniform random choice
        np.random.seed(42)
        chosen_indices = np.random.choice(len(unique_indices), MAX_POINTS_CAP, replace=False)
        unique_indices = unique_indices[chosen_indices]

    down_coords = global_coords[unique_indices].astype(np.float64) # Force float64 for Open3D
    down_colors = global_colors[unique_indices]
    down_segments = global_segments[unique_indices]
    down_instances = global_instances[unique_indices]
    
    print(f"Points remaining after downsampling: {len(down_coords)}")
    
    # Estimate Normals
    print("Estimating normals using pure NumPy/SciPy (avoiding Open3D segfaults)...")
    
    # We use cKDTree and numpy SVD for normal estimation to bypass Open3D completely
    try:
        tree_normals = cKDTree(down_coords)
        # Find 10 nearest neighbors for each point
        _, nn_indices = tree_normals.query(down_coords, k=10)
        
        down_normals = np.zeros_like(down_coords, dtype=np.float32)
        
        # Calculate normals in chunks to avoid massive memory usage
        chunk_size = 50000
        for i in tqdm(range(0, len(down_coords), chunk_size), desc="Calculating normals"):
            end_idx = min(i + chunk_size, len(down_coords))
            chunk_nn = nn_indices[i:end_idx]
            
            # Extract points for each neighborhood: shape (chunk_size, 10, 3)
            neighbors = down_coords[chunk_nn]
            
            # Center the neighbors
            centroids = np.mean(neighbors, axis=1, keepdims=True)
            centered = neighbors - centroids
            
            # Compute covariance matrices: shape (chunk_size, 3, 3)
            covariances = np.matmul(centered.transpose(0, 2, 1), centered)
            
            # Use numpy's eigh (eigenvalues/eigenvectors for Hermitian/symmetric matrices)
            # which is faster and more stable than SVD for covariance matrices
            eigenvalues, eigenvectors = np.linalg.eigh(covariances)
            
            # The normal is the eigenvector corresponding to the smallest eigenvalue
            # eigh returns eigenvalues in ascending order, so index 0 is the smallest
            normals_chunk = eigenvectors[:, :, 0]
            
            # Orient normals towards the camera (approximate origin [0,0,0])
            # dot product of normal and ray from origin to point
            pts_chunk = down_coords[i:end_idx]
            dots = np.sum(normals_chunk * pts_chunk, axis=1)
            # If dot product is positive, the normal points away from camera, so flip it
            flip_mask = dots > 0
            normals_chunk[flip_mask] = -normals_chunk[flip_mask]
            
            down_normals[i:end_idx] = normals_chunk.astype(np.float32)
            
    except Exception as e:
        print(f"Warning: Pure NumPy normal estimation failed ({e}). Filling with zeros.")
        down_normals = np.zeros_like(down_coords, dtype=np.float32)
    
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
