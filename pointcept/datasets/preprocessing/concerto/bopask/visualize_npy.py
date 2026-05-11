import os
import argparse
import numpy as np
import open3d as o3d
import glob

def main():
    parser = argparse.ArgumentParser(description="Convert npy point clouds to ply for visualization")
    parser.add_argument("--dataset_root", required=True, help="Path to preprocessed handal dataset (contains scene_* folders)")
    args = parser.parse_args()

    # Find all scene directories
    scene_dirs = glob.glob(os.path.join(args.dataset_root, "scene_*"))
    
    if not scene_dirs:
        print(f"No scene directories found in {args.dataset_root}")
        return

    for scene_dir in scene_dirs:
        coord_file = os.path.join(scene_dir, "coord.npy")
        color_file = os.path.join(scene_dir, "color.npy")
        
        if not os.path.exists(coord_file) or not os.path.exists(color_file):
            print(f"Skipping {scene_dir}, missing coord.npy or color.npy")
            continue
            
        print(f"Loading {scene_dir}...")
        coords = np.load(coord_file)
        colors = np.load(color_file)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # Open3D expects colors in range [0, 1], but color.npy is usually uint8 [0, 255]
        if colors.dtype == np.uint8 or colors.max() > 1.0:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            
        out_file = f"{scene_dir}_viz.ply"
        o3d.io.write_point_cloud(out_file, pcd)
        print(f"Saved visualization to {out_file} ({len(coords)} points)")

if __name__ == "__main__":
    main()