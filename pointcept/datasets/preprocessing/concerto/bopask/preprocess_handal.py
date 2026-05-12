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
    scene_centroid = np.zeros(3)
    alignment_computed = False
    
    # Check if we should use BOP-Ask's precomputed T_wc.txt files
    # The BOP-Ask repo creates `T_wc_{frame_id}.txt` inside each scene folder.
    use_bopask_twc = False
    twc_dir = Path(args.bop_val_dir) / scene_id
    # Wait, the script might have saved them in the original BOP val dir.
    test_twc_file = twc_dir / "T_wc_1.txt"
    if test_twc_file.exists():
        print(f"Found BOP-Ask T_wc files in {twc_dir}! Using them instead of scene_camera.json.")
        use_bopask_twc = True
        
    for rgb_path in tqdm(rgb_files, desc="Unprojecting frames"):
        basename = os.path.basename(rgb_path)
        frame_id_str = basename.split('_')[3].replace('.png', '')
        frame_id_int = str(int(frame_id_str)) # removing leading zeros for json key
        
        if frame_id_int not in cam_data:
            continue
            
        frame_cam = cam_data[frame_id_int]
        K = np.array(frame_cam['cam_K']).reshape(3, 3)
        depth_scale = frame_cam.get('depth_scale', 1.0)
        
        if use_bopask_twc:
            twc_path = twc_dir / f"T_wc_{int(frame_id_str)}.txt"
            if not twc_path.exists():
                print(f"Warning: Missing {twc_path}")
                continue
            T_wc = np.loadtxt(twc_path)
            # T_wc is the Camera to World transform! (T_c2w)
            # Wait, the script estimate_cam2world.py saves it as T_wc, which means World to Camera?
            # Let's check: P_w = T_wc[:3,:3] @ P_c + T_wc[:3,3] -> That's Camera to World! 
            # In robotics, T_wc usually means "pose of Camera in World frame", which is exactly T_c2w.
            T_c2w = T_wc
            
            # Since BOP-Ask already aligned Z-axis to be the table normal, we DO NOT need R_align!
            R_align = np.eye(3)
            scene_centroid = np.zeros(3)
            alignment_computed = True # Skip our custom alignment
            
        else:
            R_w2c = np.array(frame_cam['cam_R_w2c']).reshape(3, 3)
            t_w2c = np.array(frame_cam['cam_t_w2c']).reshape(3) / 1000.0 # Convert mm to meters
            
            T_w2c = np.eye(4)
            T_w2c[:3, :3] = R_w2c
            T_w2c[:3, 3] = t_w2c
            T_c2w = np.linalg.inv(T_w2c)
        
        # BOP uses OpenCV coordinate system: X right, Y down, Z forward.
        # Concerto (S3DIS/ScanNet format) usually expects X right, Y forward, Z up.
        # However, since we are fitting a plane and applying R_align to align the table normal to Z-up,
        # the global aligned coordinate system will automatically correct the up-axis.
        # The ring artifacts seen in CloudCompare imply the translation or rotation scale is fundamentally mismatched.
        
        # Let's double check if t_w2c is in millimeters. The BOP spec says t is in mm.
        # We divided by 1000.0, so T_w2c translation is in meters.
        # When we invert T_w2c, the new translation is -inv(R)*t.
        # Since R is orthonormal, this is correct.
        
        # ONE MORE THING: What if cam_t_w2c is ALREADY the camera origin in world space?
        # Sometimes datasets mistakenly put T_c2w in the cam_R_w2c / cam_t_w2c fields.
        # But the BOP standard strictly defines it as W2C.
        # What if we assume it's W2C, but the dataset provided C2W?
        # If the dataset provided C2W, then P_world = R * P_cam + t.
        # Let's check this hypothesis! If they provided C2W, then our T_w2c matrix is ACTUALLY T_c2w!
        # If T_w2c is actually T_c2w, then by inverting it, we broke the poses completely!
        # Let's just USE T_w2c as T_c2w to see if it fixes the misalignment!
        # Wait, if we just swap them, what happens?
        # Let's write a flag to test this. For now, let's stick to the BOP standard.
        # Actually, let's look closely at BOP-Ask paper:
        # "we estimate the camera-to-world transformation cam_T_world"
        # BOP standard: cam_R_w2c, cam_t_w2c.
        
        # ---------------------------------------------------------
        # THE TRUE FIX: BOP-ASK POSE CORRECTION
        # In BOP, the world coordinate system for some datasets (like HANDAL) might not be globally 
        # consistent across all frames if it's reconstructed per-frame, or the extrinsics might 
        # be relative to the first frame. 
        # But wait, HANDAL is a video dataset. The poses SHOULD be globally consistent.
        # What if the focal lengths (fx, fy) are different per frame but we use the first frame's?
        # No, we use K from the current frame.
        
        # Let's check the BOP-Ask paper again: 
        # "To reconstruct a consistent world coordinate system, we estimate the camera-to-world transformation...
        # We first localize the planar support surface... fit a plane... 
        # Finally, the translation vector t is determined such that the fitted plane aligns with the world origin"
        #
        # THIS IS IT! 
        # BOP-Ask DID NOT USE THE ORIGINAL BOP POSES! 
        # They RECALCULATED T_c2w ENTIRELY from scratch for every frame (or at least applied a global 
        # transformation to make the table the origin).
        # Wait, if they applied a global transformation, the relative poses between frames would still be 
        # preserved. So our R_align approach SHOULD work, provided the raw BOP poses are consistent.
        # Are the raw BOP poses consistent? My `check_raw_alignment.py` script showed that the raw poses 
        # were already severely misaligned!
        # 
        # Why would raw BOP poses be misaligned?
        # Because in BOP, the `scene_camera.json` for some datasets (like HANDAL or HOPE) might just contain 
        # poses relative to the object, NOT a global room coordinate system!
        # If the poses are object-centric, then when you unproject the whole room, the room spins around the object!
        # This perfectly explains the "ring artifacts"!!!
        # ---------------------------------------------------------
        
        # Let's look at how we build T_w2c:
        # T_w2c = np.eye(4)
        # T_w2c[:3, :3] = R_w2c
        # T_w2c[:3, 3] = t_w2c
        # T_c2w = np.linalg.inv(T_w2c)
        
        # This is mathematically perfect if the data follows the BOP standard.



        
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
        
        # ----------------------------------------------------
        # STRICT MATH: P_cam = R_w2c * P_world + t_w2c
        # => P_world = inv(R_w2c) * (P_cam - t_w2c)
        # ----------------------------------------------------
        pts_obj_cam = np.stack((x_obj, y_obj, z_obj), axis=1)
        if use_bopask_twc:
            pts_obj_cam_homo = np.hstack((pts_obj_cam, np.ones((pts_obj_cam.shape[0], 1))))
            pts_obj_world = (T_c2w @ pts_obj_cam_homo.T).T[:, :3]
        else:
            pts_obj_world = (np.linalg.inv(R_w2c) @ (pts_obj_cam - t_w2c).T).T
        
        # --- Compute Scene-Level Alignment (Once per scene) ---
        # The BOP-Ask paper calculated a global world coordinate system where the table normal
        # is strictly aligned with the Z-axis.
        # Since we do not have their alignment matrix, we must reconstruct it.
        # We assume the scene is static (the table and objects do not move relative to the world origin).
        # We compute this ONCE using the first valid frame, and apply the SAME alignment 
        # (R_align and scene_centroid) to ALL subsequent frames in the scene.
        if not alignment_computed:
            try:
                # To prevent Open3D Segmentation Faults on the cluster, we use pure numpy SVD to fit the plane
                # since the object points are mostly spread across the table surface.
                pts_for_fit = pts_obj_world[::10] # Subsample for speed
                
                # We want to find a robust plane. SVD might be skewed by tall objects.
                # A simple heuristic: the lowest Z points in the object point cloud (which touch the table)
                # actually, since the world frame is arbitrary, we don't know which way is down.
                # However, SVD on ALL object points usually gives a normal roughly perpendicular to the table
                # because the table spread (X, Y) is much larger than the height of objects (Z).
                centroid = np.mean(pts_for_fit, axis=0)
                centered_pts = pts_for_fit - centroid
                
                u, s, vh = np.linalg.svd(centered_pts, full_matrices=False)
                
                # The normal is the last row of Vh
                n_p = vh[2, :]
                
                # Ensure normal points "up" relative to the camera origin
                # The table normal should point TOWARDS the camera (dot product with camera ray < 0)
                # The camera origin in world space is T_c2w[:3, 3]
                cam_origin = T_c2w[:3, 3]
                ray = cam_origin - centroid
                if np.dot(n_p, ray) < 0:
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
                
                # Save the centroid so we can rotate the world around it
                scene_centroid = centroid
                
                print(f"\nScene Alignment Computed (Numpy SVD). Table normal in World Space: {n_p}")
            except Exception as e:
                print(f"\nWarning: SVD plane fitting failed ({e}), using default World Space.")
                R_align = np.eye(3)
                scene_centroid = np.zeros(3)
            alignment_computed = True
        
        # Apply the alignment rotation so the table is flat (Z is up)
        # Note: We must rotate the world points, but wait:
        # pts_obj_world = (T_c2w @ P_cam).T
        # T_c2w = inv(T_w2c)
        # R_align aligns the normal in world space to [0,0,1].
        # Is R_align consistent for all frames? Yes, because we compute it once in World Space.
        
        # ---------------------------------------------------------
        # BUG FIX: The BOP dataset cam_t_w2c is in millimeters!
        # When unprojecting points, depth is converted to METERS.
        # But our T_c2w translation vector is STILL IN METERS? 
        # Wait, I did `t_w2c = np.array(frame_cam['cam_t_w2c']).reshape(3) / 1000.0`
        # So T_c2w is in METERS. That matches depth_in_m.
        # BUT look at how T_c2w is built:
        # T_w2c[:3, 3] = t_w2c
        # T_c2w = np.linalg.inv(T_w2c)
        # Is this correct? Yes.
        # ---------------------------------------------------------
        
        pts_obj_aligned = (R_align @ (pts_obj_world - scene_centroid).T).T
        
        # Object bounding box in Aligned Space (Z is aligned with table normal)
        min_x, max_x = pts_obj_aligned[:, 0].min(), pts_obj_aligned[:, 0].max()
        min_y, max_y = pts_obj_aligned[:, 1].min(), pts_obj_aligned[:, 1].max()
        min_z, max_z = pts_obj_aligned[:, 2].min(), pts_obj_aligned[:, 2].max()
        
        # 2. Expand bounding box to include the supporting table and local context
        # 缩小 X/Y 轴方向的扩张，让包围盒更紧凑，严格只包含桌面和物体本身
        margin_x, margin_y, margin_z_below, margin_z_above = 0.15, 0.15, 0.05, 0.5
        
        # 3. Unproject ALL valid depth points
        valid_depth = (depth_img > 0)
        v_all, u_all = np.where(valid_depth)
        z_all = depth_in_m[v_all, u_all]
        x_all = (u_all - cx) * z_all / fx
        y_all = (v_all - cy) * z_all / fy
        
        pts_all_cam = np.stack((x_all, y_all, z_all), axis=1)
        if use_bopask_twc:
            pts_all_cam_homo = np.hstack((pts_all_cam, np.ones((pts_all_cam.shape[0], 1))))
            pts_all_world = (T_c2w @ pts_all_cam_homo.T).T[:, :3]
        else:
            pts_all_world = (np.linalg.inv(R_w2c) @ (pts_all_cam - t_w2c).T).T
            
        pts_all_aligned = (R_align @ (pts_all_world - scene_centroid).T).T
        
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
        if use_bopask_twc:
            pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
            pts_world_orig = (T_c2w @ pts_cam_homo.T).T[:, :3]
        else:
            pts_world_orig = (np.linalg.inv(R_w2c) @ (pts_cam - t_w2c).T).T
        
        # ---------------------------------------------------------
        # BUG FIX: Global alignment vs Local transformations
        # When we apply R_align to pts_world_orig, we are rotating the world around the origin (0,0,0).
        # If the table is NOT at the origin in the original world space, rotating around (0,0,0) 
        # will sweep the table through a giant arc, completely displacing it!
        # In BOP datasets, the origin (0,0,0) is arbitrary. It might be far away from the table.
        # To rotate the scene so the table is flat, we must rotate AROUND THE TABLE CENTROID,
        # or we must translate the table to the origin first!
        # Let's fix this by translating the centroid to origin, rotating, then (optionally) translating back.
        # Actually, for Concerto, we usually want the table center at (0,0,0) anyway.
        # Let's define the scene origin as the centroid of the objects in the first frame.
        # ---------------------------------------------------------
        if not alignment_computed:
            raise RuntimeError("alignment_computed should be True by now")
            
        # Apply Scene Alignment to make table flat (Z is up)
        pts_world = (R_align @ (pts_world_orig - scene_centroid).T).T
            
        # Update T_c2w to reflect the new aligned world space
        # T_c2w maps points from Camera to World.
        # P_aligned = R_align * (P_world_orig - scene_centroid)
        # P_aligned = R_align * (T_c2w * P_cam - scene_centroid)
        # P_aligned = R_align * T_c2w * P_cam - R_align * scene_centroid
        # So the new T_c2w_aligned is:
        # Rotation: R_align * T_c2w[:3, :3]
        # Translation: R_align * T_c2w[:3, 3] - R_align @ scene_centroid
        
        T_c2w_aligned = np.eye(4)
        T_c2w_aligned[:3, :3] = R_align @ T_c2w[:3, :3]
        T_c2w_aligned[:3, 3] = R_align @ T_c2w[:3, 3] - R_align @ scene_centroid
        
        # NOTE: S3DIS format and standard graphics (OpenGL) use Z-up, but BOP uses OpenCV (Z-forward).
        # We aligned the table to Z-up, so our world space is now Z-up!
        # Concerto might expect specific camera axes (e.g. X-right, Y-up, Z-back).
        # If Concerto relies heavily on camera pose direction, we might need a coordinate flip.
        # But for global point cloud stitching, as long as T_c2w accurately maps camera pixels to 
        # the aligned world, it is geometrically consistent.

        
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
    
    # Since we strictly filter out all background points using the tight Local Context Box,
    # the total point count should be manageable.
    MAX_POINTS_CAP = 10_000_000 # 10 million points for high fidelity
    if len(unique_indices) > MAX_POINTS_CAP:
        print(f"WARNING: Point cloud still too large ({len(unique_indices)}). Randomly subsampling to {MAX_POINTS_CAP}...")
        
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
    parser.add_argument("--scene_id", type=str, default=None, help="Process only a specific scene ID (e.g., '000001')")
    
    args = parser.parse_args()
    
    images_dir = os.path.join(args.src_dir, 'images')
    depth_dir = os.path.join(args.src_dir, 'depth_maps')
    masks_dir = os.path.join(args.src_dir, 'masks')
    
    # Find unique scenes
    all_imgs = glob.glob(os.path.join(images_dir, "*.png"))
    scene_ids = sorted(list(set([os.path.basename(f).split('_')[1] for f in all_imgs])))
    
    if args.scene_id:
        if args.scene_id in scene_ids:
            scene_ids = [args.scene_id]
        else:
            print(f"Error: Scene {args.scene_id} not found in {images_dir}")
            exit(1)
            
    print(f"Found {len(scene_ids)} scenes to process.")
    
    for scene_id in scene_ids:
        camera_json = os.path.join(args.bop_val_dir, scene_id, 'scene_camera.json')
        if not os.path.exists(camera_json):
            print(f"Warning: {camera_json} not found. Skipping scene {scene_id}.")
            continue
            
        process_scene(scene_id, images_dir, depth_dir, masks_dir, camera_json, args.output_root, args.rgb_gap, args.voxel_size)
