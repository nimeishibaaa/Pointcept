import os
import sys
import json
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

def save_ply_manual(pts, colors, filepath):
    with open(filepath, "wb") as f:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        f.write(header.encode('ascii'))
        data = np.empty(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', '<u1'), ('g', '<u1'), ('b', '<u1')])
        data['x'] = pts[:, 0]
        data['y'] = pts[:, 1]
        data['z'] = pts[:, 2]
        data['r'] = colors[:, 0]
        data['g'] = colors[:, 1]
        data['b'] = colors[:, 2]
        f.write(data.tobytes())

def get_point_cloud_from_rgbd(rgb_img, depth_img, masks_dict, K, R_w2c, t_w2c, depth_scale=1000.0):
    H, W = depth_img.shape
    
    depth_m = depth_img * depth_scale / 1000.0
    v_all, u_all = np.where(depth_m > 0)
    if len(v_all) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    z_all = depth_m[v_all, u_all]
    x_all = (u_all - K[0,2]) * z_all / K[0,0]
    y_all = (v_all - K[1,2]) * z_all / K[1,1]
    
    # 1. Unproject ALL points to original world coordinates
    pts_cam_all = np.stack([x_all, y_all, z_all], axis=1)
    R_inv = np.linalg.inv(R_w2c)
    pts_world_all = (pts_cam_all - t_w2c) @ R_inv.T
    
    # Semantic & Instance segments
    # Strategy C: Unknown Tabletop Objects as Obstacles.
    # Since we have perfectly cropped the scene to the local tabletop (removing all walls/distant background),
    # any unannotated points left on the table are guaranteed to be physical objects (distractors/obstacles)
    # rather than just empty space or distant walls.
    # To enhance robotic manipulation safety, we explicitly label these unknown tabletop objects
    # as a distinct "obstacle" class (e.g., 41), forcing the model to actively segment them from the table,
    # rather than ignoring them (-1) which might lead the robot to crash into them.
    # Note: We still label the actual flat table surface as background (0) if we can identify it,
    # but for simplicity without table masks, we label all unannotated points as 0 (Background/Obstacle).
    # Wait, the user wants explicit "obstacle" representation for unknown objects.
    # Let's initialize everything to 0 (Background/Obstacle class).
    segment_full = np.zeros((H, W), dtype=np.int32)
    instance_full = np.zeros((H, W), dtype=np.int32)
    has_objects = np.zeros((H, W), dtype=bool)
    
    for class_id, mask in masks_dict.items():
        valid = mask > 0
        segment_full[valid] = class_id
        instance_full[valid] = class_id
        has_objects[valid] = True
        
    # 2. Extract ONLY the object points to compute Alignment and Bounding Box
    v_obj, u_obj = np.where(has_objects & (depth_m > 0))
    if len(v_obj) < 10: # Too few points to fit a plane
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    z_obj = depth_m[v_obj, u_obj]
    x_obj = (u_obj - K[0,2]) * z_obj / K[0,0]
    y_obj = (v_obj - K[1,2]) * z_obj / K[1,1]
    pts_cam_obj = np.stack([x_obj, y_obj, z_obj], axis=1)
    pts_world_obj = (pts_cam_obj - t_w2c) @ R_inv.T
    
    # 3. Compute Scene Alignment (Z-up based on Table Surface)
    # Instead of fitting a plane to the objects (which causes tilt if objects are tall/asymmetric),
    # we find the dominant plane (table) in the local neighborhood using robust RANSAC.
    centroid = np.mean(pts_world_obj, axis=0)
    
    # Gather ALL points within 0.5m of the object centroid (this captures the table)
    dists = np.linalg.norm(pts_world_all - centroid, axis=1)
    local_pts = pts_world_all[dists < 0.5]
    if len(local_pts) < 100:
        local_pts = pts_world_obj
        
    def ransac_normal(pts, iters=1000, thresh=0.005):
        if len(pts) < 3: return np.array([0.0, 0.0, 1.0])
        idx = np.random.randint(0, len(pts), (iters, 3))
        p1, p2, p3 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        valid = norm[:, 0] > 1e-6
        if not np.any(valid): return np.array([0.0, 0.0, 1.0])
        n, p1 = n[valid] / norm[valid], p1[valid]
        
        eval_pts = pts[::max(1, len(pts)//5000)] # Subsample for fast evaluation
        
        # Vectorized distance computation to replace the slow for-loop
        d = -np.sum(n * p1, axis=1) # (K,)
        dists = np.abs(eval_pts @ n.T + d) # (N, K)
        inliers_count = np.sum(dists < thresh, axis=0) # (K,)
        best_idx = np.argmax(inliers_count)
        
        return n[best_idx]

    n_p = ransac_normal(local_pts)
        
    # Orient normal towards camera
    C_world = -R_inv @ t_w2c
    ray_to_cam = C_world - centroid
    if np.dot(n_p, ray_to_cam) < 0:
        n_p = -n_p
        
    # Compute R_align to rotate n_p to [0, 0, 1]
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
        
    # 4. Align object points to get Bounding Box
    # Use percentiles instead of absolute min/max to ignore "flying pixels" (mask bleeding onto distant background)
    pts_obj_aligned = (pts_world_obj - centroid) @ R_align.T
    min_x, max_x = np.percentile(pts_obj_aligned[:, 0], [0.5, 99.5])
    min_y, max_y = np.percentile(pts_obj_aligned[:, 1], [0.5, 99.5])
    min_z, max_z = np.percentile(pts_obj_aligned[:, 2], [0.5, 99.5])
    
    # 5. Expand bounding box to include local tabletop
    margin_x, margin_y, margin_z_below, margin_z_above = 0.15, 0.15, 0.03, 0.5
    
    # 6. Align ALL points and filter by expanded Bounding Box
    pts_all_aligned = (pts_world_all - centroid) @ R_align.T
    in_box = (
        (pts_all_aligned[:, 0] >= min_x - margin_x) & (pts_all_aligned[:, 0] <= max_x + margin_x) &
        (pts_all_aligned[:, 1] >= min_y - margin_y) & (pts_all_aligned[:, 1] <= max_y + margin_y) &
        (pts_all_aligned[:, 2] >= min_z - margin_z_below) & (pts_all_aligned[:, 2] <= max_z + margin_z_above)
    )
    
    pts_valid_aligned = pts_all_aligned[in_box]
    if len(pts_valid_aligned) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    # 7. Compute Camera Ray Prior in the ALIGNED coordinate system
    C_aligned = (C_world - centroid) @ R_align.T
    ray_aligned = pts_valid_aligned - C_aligned
    ray_norm = np.linalg.norm(ray_aligned, axis=1, keepdims=True)
    ray_norm[ray_norm == 0] = 1e-6
    ray_aligned = ray_aligned / ray_norm
    
    # Extract valid attributes
    color_valid = rgb_img[v_all[in_box], u_all[in_box]]
    segment_valid = segment_full[v_all[in_box], u_all[in_box]]
    instance_valid = instance_full[v_all[in_box], u_all[in_box]]
    
    # 8. Heuristic Separation of "Table Surface" (0) and "Unknown Obstacles" (41)
    # Since pts_valid_aligned has Z-axis strictly pointing up from the table,
    # the table surface is roughly at Z = min_z.
    # Any unannotated point (segment == 0) that is significantly higher than the table
    # (e.g., > 1.5 cm) is a physical obstacle (like a box or first-aid kit).
    unannotated_mask = (segment_valid == 0)
    obstacle_z_threshold = min_z + 0.01  # 1.0 cm above the lowest object point
    is_obstacle = unannotated_mask & (pts_valid_aligned[:, 2] > obstacle_z_threshold)
    
    segment_valid[is_obstacle] = 41
    instance_valid[is_obstacle] = 41
    
    return pts_valid_aligned, color_valid, segment_valid, instance_valid, ray_aligned

def generate_semantic_colors(segment_array):
    """
    Generate distinct colors for semantic visualization based on BOPAsk classes.
    0: background (light gray)
    1-9: hammer (red)
    10-14: spatula (green)
    15-19: measuring spoon (blue)
    20-26: power drill (yellow)
    27-30: ladle (cyan)
    31-34: strainer (magenta)
    35-40: whisk (orange)
    41: obstacle (purple)
    """
    colors = np.zeros((len(segment_array), 3), dtype=np.uint8)
    
    # Map class IDs to categories using fast numpy boolean indexing
    colors[segment_array == 0] = [200, 200, 200]
    colors[(segment_array >= 1) & (segment_array <= 9)] = [255, 0, 0]
    colors[(segment_array >= 10) & (segment_array <= 14)] = [0, 255, 0]
    colors[(segment_array >= 15) & (segment_array <= 19)] = [0, 0, 255]
    colors[(segment_array >= 20) & (segment_array <= 26)] = [255, 255, 0]
    colors[(segment_array >= 27) & (segment_array <= 30)] = [0, 255, 255]
    colors[(segment_array >= 31) & (segment_array <= 34)] = [255, 0, 255]
    colors[(segment_array >= 35) & (segment_array <= 40)] = [255, 165, 0]
    colors[segment_array == 41] = [128, 0, 128]
            
    return colors

def parse_bopask(src_dir, out_dir, bop_original_dir, split_type, debug=False):
    os.makedirs(out_dir, exist_ok=True)
    images = glob.glob(os.path.join(src_dir, 'images', '*.png'))
    # Filter augmented images
    import re
    valid_pattern = re.compile(r'scene_\d+_frame_\d+\.png$')
    images = [img for img in images if valid_pattern.match(os.path.basename(img))]
    # CRITICAL: Sort images so processing order is deterministic for resume capability
    images = sorted(images)
    print(f"Found {len(images)} valid raw images to process in {src_dir}.")
    
    # Group by scene
    scene_dict = {}
    for img_path in images:
        basename = os.path.basename(img_path)
        scene_id = basename.split('_')[1]
        if scene_id not in scene_dict:
            scene_dict[scene_id] = []
        scene_dict[scene_id].append(img_path)
        
    # We want to process ONLY scene_000001, frame 000001 and frame 000002 to debug the projection issue
    if debug:
        print("DEBUG: scene_000001 frame_000001 for vis")
        scene_dict = {'000001': [p for p in scene_dict.get('000001', []) if 'frame_000001.png' in p]}
        
    # Sort scenes dictionary by keys to ensure deterministic processing order
    scene_dict = dict(sorted(scene_dict.items()))
    
    pbar = tqdm(scene_dict.items(), desc="Processing scenes")
    for scene_id, img_list in pbar:
        pbar.set_postfix({'scene': scene_id})
        # BOPAsk images are derived from original BOP scenes
        # Use user-specified split_type (e.g., 'val' or 'test') to locate the camera json
        cam_json_path = os.path.join(bop_original_dir, split_type, scene_id, 'scene_camera.json')
            
        if not os.path.exists(cam_json_path):
            # Fallback if camera json is missing
            continue
            
        with open(cam_json_path, 'r') as f:
            cam_data = json.load(f)
            
        scene_out_dir = os.path.join(out_dir, f"scene_{scene_id}")
        os.makedirs(scene_out_dir, exist_ok=True)
            
        for img_path in img_list:
            basename = os.path.basename(img_path)
            name_stem = basename.replace('.png', '')
            frame_id_str = str(int(name_stem.split('_')[3]))
            
            # Check for existing processed data to support resuming
            sample_dir = os.path.join(scene_out_dir, name_stem)
            expected_files = ['coord.npy', 'color.npy', 'segment.npy', 'instance.npy', 'normal.npy']
            if not debug and all(os.path.exists(os.path.join(sample_dir, f)) for f in expected_files):
                continue
            
            if frame_id_str not in cam_data:
                continue
                
            cam = cam_data[frame_id_str]
            K = np.array(cam['cam_K']).reshape(3,3)
            R_w2c = np.array(cam['cam_R_w2c']).reshape(3,3)
            t_w2c = np.array(cam['cam_t_w2c']).reshape(3) / 1000.0
            # BOP datasets (like HANDAL) often use a depth scale of 0.1 mm, not 1.0 mm.
            # If missing from json, default to 0.1 to prevent severe depth distortion (non-planar floors).
            depth_scale = cam.get('depth_scale', 0.1)
            
            depth_path = os.path.join(src_dir, 'depth_maps', basename.replace('.png', '_depth.png'))
            if not os.path.exists(depth_path):
                depth_path = os.path.join(src_dir, 'depth_maps', basename)
            if not os.path.exists(depth_path):
                continue
                
            mask_pattern = os.path.join(src_dir, 'masks', f"{name_stem}*_mask.png")
            mask_files = glob.glob(mask_pattern)
            masks_dict = {}
            for mf in mask_files:
                parts = os.path.basename(mf).split('_')
                try:
                    idx = parts.index('mask.png')
                    class_id = int(parts[idx-1])
                    mask_img = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                    masks_dict[class_id] = mask_img
                except:
                    continue
                    
            rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            coord, color, segment, instance, ray = get_point_cloud_from_rgbd(
                rgb_img, depth_img, masks_dict, K, R_w2c, t_w2c, depth_scale
            )
            
            if len(coord) == 0:
                continue
                
            # Pointcept saves each frame as a folder in some formats, or just directly in scene folder.
            # Here we save per frame, mimicking Concerto's chunks
            sample_dir = os.path.join(scene_out_dir, name_stem)
            os.makedirs(sample_dir, exist_ok=True)
            
            np.save(os.path.join(sample_dir, 'coord.npy'), coord.astype(np.float32))
            np.save(os.path.join(sample_dir, 'color.npy'), color.astype(np.uint8))
            np.save(os.path.join(sample_dir, 'segment.npy'), segment.astype(np.int16))
            np.save(os.path.join(sample_dir, 'instance.npy'), instance.astype(np.int16))
            # *CRITICAL*: Save camera ray into normal.npy so Pointcept automatically loads it
            np.save(os.path.join(sample_dir, 'normal.npy'), ray.astype(np.float32))

            vis_dir = os.path.join(out_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            if debug:
                ply_path = os.path.join(vis_dir, f"{name_stem}_debug.ply")
                save_ply_manual(coord, color, ply_path)
                
                # Visualize Camera Ray by mapping [-1, 1] to RGB [0, 255]
                ray_ply_path = os.path.join(vis_dir, f"{name_stem}_debug_ray.ply")
                ray_colors = ((ray + 1.0) * 127.5).astype(np.uint8)
                save_ply_manual(coord, ray_colors, ray_ply_path)
                
                # Visualize Semantic Segmentation based on PPT categories
                sem_ply_path = os.path.join(vis_dir, f"{name_stem}_debug_semantic.ply")
                sem_colors = generate_semantic_colors(segment)
                save_ply_manual(coord, sem_colors, sem_ply_path)
                
                print(f"\n[DEBUG MODE] Processed one image: {name_stem}")
                print(f" - Saved numpy arrays to {sample_dir}")
                print(f" - Saved RGB PLY point cloud to {ply_path}")
                print(f" - Saved Ray visualization PLY to {ray_ply_path}")
                print(f" - Saved Semantic visualization PLY to {sem_ply_path}")
                sys.exit(0)
            else:
                # Save 2x downsampled sparse PLY for every view in non-debug mode for robust logging/debugging
                ply_path = os.path.join(vis_dir, f"{name_stem}_sparse.ply")
                save_ply_manual(coord[::2], color[::2], ply_path)
                
                ray_ply_path = os.path.join(vis_dir, f"{name_stem}_sparse_ray.ply")
                ray_colors = ((ray[::2] + 1.0) * 127.5).astype(np.uint8)
                save_ply_manual(coord[::2], ray_colors, ray_ply_path)
                
                sem_ply_path = os.path.join(vis_dir, f"{name_stem}_sparse_semantic.ply")
                sem_colors = generate_semantic_colors(segment[::2])
                save_ply_manual(coord[::2], sem_colors, sem_ply_path)

            # Also generate a pseudo text mapping file if using PPT open-vocabulary
            # Handal classes are just target_X, so we use dummy "obj_X" for now.
            # In a real PPT setting, you'd map class_id to a real English name (e.g. 1 -> "mug")
            # to compute the CLIP text embedding.
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Path to BOPAsk images/depths (e.g., .../bopask-data/handal-qa-train)")  # BOPAsk训练集来自BOP挑战赛验证集的十个场景
    parser.add_argument("--out_dir", type=str, required=True, help="Path to output Pointcept data (e.g., .../Pointcept/data/concerto/bopask)")
    parser.add_argument("--bop_original_dir", type=str, required=True, help="Path to original BOP dataset containing camera intrinsics/extrinsics (e.g., .../bop_original/handal)")  # BOP挑战赛原始数据
    parser.add_argument("--split_type", type=str, default="val", choices=["val", "test", "train"], help="Which split in bop_original_dir contains the scene_camera.json for these images (usually 'val' or 'test')")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode: process 1 image, save PLY, and exit.")
    args = parser.parse_args()
    parse_bopask(args.src_dir, args.out_dir, args.bop_original_dir, args.split_type, args.debug)
