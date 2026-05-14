import json
import numpy as np
import cv2
import os
import open3d as o3d

scene_id = "000001"
bop_dir = "/gpfs/work/mec/yiwenwang18/datasets/bop_original/handal/val"
src_dir = "/gpfs/work/mec/yiwenwang18/datasets/bopask-data/handal-qa-train"

with open(os.path.join(bop_dir, scene_id, 'scene_camera.json'), 'r') as f:
    cam_data = json.load(f)

# Pick three frames that definitely exist in scene 000001
available_frames = list(cam_data.keys())
frames = [available_frames[0], available_frames[len(available_frames)//2], available_frames[-1]]
print(f"Testing with frames: {frames}")

all_pts = []
all_colors = []

for frame_id in frames:
    cam = cam_data[frame_id]
    K = np.array(cam['cam_K']).reshape(3,3)
    R_w2c = np.array(cam['cam_R_w2c']).reshape(3,3)
    t_w2c = np.array(cam['cam_t_w2c']).reshape(3) / 1000.0  # mm to meters
    depth_scale = cam.get('depth_scale', 1.0)
    
    depth_path = os.path.join(src_dir, 'depth_maps', f'scene_{scene_id}_frame_0000{frame_id.zfill(2)}_depth.png')
    rgb_path = os.path.join(src_dir, 'images', f'scene_{scene_id}_frame_0000{frame_id.zfill(2)}.png')
    if not os.path.exists(depth_path): continue
        
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth_m = depth * depth_scale / 1000.0
    v, u = np.where(depth_m > 0)
    # v, u = v[::5], u[::5]
    z = depth_m[v, u]
    x = (u - K[0,2]) * z / K[0,0]
    y = (v - K[1,2]) * z / K[1,1]
    pts_cam = np.stack([x, y, z], axis=1)
    
    # ----------------------------------------------------
    # STRICT MATH: P_cam = R_w2c * P_world + t_w2c
    # => P_world = inv(R_w2c) * (P_cam - t_w2c)
    # ----------------------------------------------------
    pts_world = (np.linalg.inv(R_w2c) @ (pts_cam - t_w2c).T).T
    all_pts.append(pts_world)
    all_colors.append(rgb[v, u])

print("Finished unprojecting. Saving PLY without Open3D...")

# Combine all points and colors
final_pts = np.vstack(all_pts)
final_colors = np.vstack(all_colors)

# Save to PLY manually to completely avoid Open3D segfaults
with open("/gpfs/work/mec/yiwenwang18/raw_strict_test.ply", "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(final_pts)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for i in range(len(final_pts)):
        p = final_pts[i]
        c = final_colors[i]
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

print("Saved to /gpfs/work/mec/yiwenwang18/raw_strict_test.ply")
