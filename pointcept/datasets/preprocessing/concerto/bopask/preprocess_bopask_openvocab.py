import os
import sys
import json
import glob
import numpy as np
import cv2
from tqdm import tqdm
import argparse

# ── 填写你的 BOP obj_id → 文本描述映射 ──────────────────────────
# 键是 BOP 数据集里的整数 obj_id（mask文件名里那个数字）
# 值是你想用于 CLIP 检索的英文文本描述
# 同一类别的不同实例（不同 obj_id）可以映射到相同文本
# 例如：{1: "hammer", 2: "hammer", 10: "spatula", ...}
OBJ_ID_TO_TEXT = {
     # 1-9: Hammers
    1: "wooden handle hammer", 2: "hammer", 3: "hammer", 4: "black handle hammer", 5: "hammer", 6: "hammer", 7: "hammer", 8: "wooden handle hammer", 9: "black handle hammer",
    # 10-14: Spoons / Ladles
    10: "green slotted spoon", 11: "red ladle", 12: "green ladle", 13: "spaghetti spoon", 
    # 14-19: Measuring spoons
    14: "small orange measuring spoon", 15: "measuring spoon", 16: "measuring spoon", 17: "yellow measuring spoon", 18: "measuring spoon", 19: "large orange measuring spoon",
    # 20-25: Power drills
    20: "green power drill", 21: "green power drill", 22: "orange power drill", 23: "orange power drill", 24: "red power drill", 25: "red power drill", 
    # 26-29: Spatulas
    26: "purple spatula", 27: "magenta spatula", 28: "orange spatula", 29: "cyan spatula", 
    # 30-34: Strainers
    30: "metal strainer", 31: "big metal strainer", 32: "strainer", 33: "strainer", 34: "green strainer",
    # 35-40: Whisks
    35: "whisk", 36: "whisk", 37: "whisk", 38: "whisk", 39: "red whisk", 40: "yellow whisk"
}

# ─────────────────────────────────────────────────────────────────

def save_ply_manual(pts, colors, filepath):
    with open(filepath, "wb") as f:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        f.write(header.encode('ascii'))
        data = np.empty(len(pts), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', '<u1'), ('g', '<u1'), ('b', '<u1')
        ])
        data['x'] = pts[:, 0]; data['y'] = pts[:, 1]; data['z'] = pts[:, 2]
        data['r'] = colors[:, 0]; data['g'] = colors[:, 1]; data['b'] = colors[:, 2]
        f.write(data.tobytes())


def get_point_cloud_from_rgbd(rgb_img, depth_img, masks_dict, K, R_w2c, t_w2c, depth_scale=0.1):
    H, W = depth_img.shape

    depth_m = depth_img * depth_scale / 1000.0
    v_all, u_all = np.where(depth_m > 0)
    if len(v_all) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    z_all = depth_m[v_all, u_all]
    x_all = (u_all - K[0, 2]) * z_all / K[0, 0]
    y_all = (v_all - K[1, 2]) * z_all / K[1, 1]

    pts_cam_all = np.stack([x_all, y_all, z_all], axis=1)
    R_inv = np.linalg.inv(R_w2c)
    pts_world_all = (pts_cam_all - t_w2c) @ R_inv.T

    segment_full = np.zeros((H, W), dtype=np.int32)
    instance_full = np.zeros((H, W), dtype=np.int32)
    has_objects = np.zeros((H, W), dtype=bool)

    for class_id, mask in masks_dict.items():
        valid = mask > 0
        segment_full[valid] = class_id
        instance_full[valid] = class_id
        has_objects[valid] = True

    v_obj, u_obj = np.where(has_objects & (depth_m > 0))
    if len(v_obj) < 10:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    z_obj = depth_m[v_obj, u_obj]
    x_obj = (u_obj - K[0, 2]) * z_obj / K[0, 0]
    y_obj = (v_obj - K[1, 2]) * z_obj / K[1, 1]
    pts_cam_obj = np.stack([x_obj, y_obj, z_obj], axis=1)
    pts_world_obj = (pts_cam_obj - t_w2c) @ R_inv.T

    centroid = np.mean(pts_world_obj, axis=0)
    dists = np.linalg.norm(pts_world_all - centroid, axis=1)
    local_pts = pts_world_all[dists < 0.5]
    if len(local_pts) < 100:
        local_pts = pts_world_obj

    def ransac_normal(pts, iters=1000, thresh=0.005):
        if len(pts) < 3:
            return np.array([0.0, 0.0, 1.0])
        idx = np.random.randint(0, len(pts), (iters, 3))
        p1, p2, p3 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        valid = norm[:, 0] > 1e-6
        if not np.any(valid):
            return np.array([0.0, 0.0, 1.0])
        n, p1 = n[valid] / norm[valid], p1[valid]
        eval_pts = pts[::max(1, len(pts) // 5000)]
        d = -np.sum(n * p1, axis=1)
        dists = np.abs(eval_pts @ n.T + d)
        inliers_count = np.sum(dists < thresh, axis=0)
        return n[np.argmax(inliers_count)]

    n_p = ransac_normal(local_pts)

    C_world = -R_inv @ t_w2c
    if np.dot(n_p, C_world - centroid) < 0:
        n_p = -n_p

    v_z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n_p, v_z)
    c = np.dot(n_p, v_z)
    if np.linalg.norm(v) < 1e-6:
        R_align = np.eye(3)
    else:
        s_norm = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / s_norm ** 2)

    pts_obj_aligned = (pts_world_obj - centroid) @ R_align.T
    min_x, max_x = np.percentile(pts_obj_aligned[:, 0], [0.5, 99.5])
    min_y, max_y = np.percentile(pts_obj_aligned[:, 1], [0.5, 99.5])
    min_z, max_z = np.percentile(pts_obj_aligned[:, 2], [0.5, 99.5])

    margin_x, margin_y, margin_z_below, margin_z_above = 0.15, 0.15, 0.03, 0.5
    pts_all_aligned = (pts_world_all - centroid) @ R_align.T
    in_box = (
        (pts_all_aligned[:, 0] >= min_x - margin_x) & (pts_all_aligned[:, 0] <= max_x + margin_x) &
        (pts_all_aligned[:, 1] >= min_y - margin_y) & (pts_all_aligned[:, 1] <= max_y + margin_y) &
        (pts_all_aligned[:, 2] >= min_z - margin_z_below) & (pts_all_aligned[:, 2] <= max_z + margin_z_above)
    )

    pts_valid_aligned = pts_all_aligned[in_box]
    if len(pts_valid_aligned) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    C_aligned = (C_world - centroid) @ R_align.T
    ray_aligned = pts_valid_aligned - C_aligned
    ray_norm = np.linalg.norm(ray_aligned, axis=1, keepdims=True)
    ray_norm[ray_norm == 0] = 1e-6
    ray_aligned = ray_aligned / ray_norm

    color_valid    = rgb_img[v_all[in_box], u_all[in_box]]
    segment_valid  = segment_full[v_all[in_box], u_all[in_box]]
    instance_valid = instance_full[v_all[in_box], u_all[in_box]]

    # 高于桌面 1cm 且无标注的点 → 障碍物(41)，后续合并回背景
    unannotated = (segment_valid == 0)
    is_obstacle = unannotated & (pts_valid_aligned[:, 2] > min_z + 0.01)
    segment_valid[is_obstacle]  = 41
    instance_valid[is_obstacle] = 41

    return pts_valid_aligned, color_valid, segment_valid, instance_valid, ray_aligned


def generate_semantic_colors(segment_array):
    """用于 debug PLY 可视化，按 obj_id 着色。"""
    rng = np.random.default_rng(0)
    palette = rng.integers(50, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = [200, 200, 200]   # 背景灰色
    palette[41] = [128, 0, 128]    # 障碍物紫色
    ids = segment_array % 256
    return palette[ids]


def parse_bopask(src_dir, out_dir, bop_original_dir, split_type, debug=False):
    os.makedirs(out_dir, exist_ok=True)

    import re
    images = sorted(glob.glob(os.path.join(src_dir, 'images', '*.png')))
    valid_pattern = re.compile(r'scene_\d+_frame_\d+\.png$')
    images = [img for img in images if valid_pattern.match(os.path.basename(img))]
    print(f"Found {len(images)} valid images in {src_dir}.")

    # 按 scene 分组
    scene_dict = {}
    for img_path in images:
        scene_id = os.path.basename(img_path).split('_')[1]
        scene_dict.setdefault(scene_id, []).append(img_path)
    scene_dict = dict(sorted(scene_dict.items()))

    if debug:
        first_scene = next(iter(scene_dict))
        scene_dict = {first_scene: scene_dict[first_scene][:1]}
        print(f"DEBUG: processing 1 frame from scene {first_scene}")

    pbar = tqdm(scene_dict.items(), desc="Processing scenes")
    for scene_id, img_list in pbar:
        pbar.set_postfix({'scene': scene_id})

        cam_json_path = os.path.join(bop_original_dir, split_type, scene_id, 'scene_camera.json')
        if not os.path.exists(cam_json_path):
            continue
        with open(cam_json_path) as f:
            cam_data = json.load(f)

        scene_out_dir = os.path.join(out_dir, f"scene_{scene_id}")
        os.makedirs(scene_out_dir, exist_ok=True)

        for img_path in img_list:
            basename    = os.path.basename(img_path)
            name_stem   = basename.replace('.png', '')
            frame_id_str = str(int(name_stem.split('_')[3]))

            sample_dir = os.path.join(scene_out_dir, name_stem)
            done_files = ['coord.npy', 'color.npy', 'segment.npy', 'normal.npy', 'labels.json']
            if not debug and all(os.path.exists(os.path.join(sample_dir, f)) for f in done_files):
                continue

            if frame_id_str not in cam_data:
                continue

            cam = cam_data[frame_id_str]
            K     = np.array(cam['cam_K']).reshape(3, 3)
            R_w2c = np.array(cam['cam_R_w2c']).reshape(3, 3)
            t_w2c = np.array(cam['cam_t_w2c']).reshape(3) / 1000.0
            depth_scale = cam.get('depth_scale', 0.1)

            # 找深度图（两种可能的文件名约定）
            depth_path = os.path.join(src_dir, 'depth_maps', basename.replace('.png', '_depth.png'))
            if not os.path.exists(depth_path):
                depth_path = os.path.join(src_dir, 'depth_maps', basename)
            if not os.path.exists(depth_path):
                continue

            # 读所有 mask：文件名含 obj_id
            mask_files = glob.glob(os.path.join(src_dir, 'masks', f"{name_stem}*_mask.png"))
            masks_dict = {}
            for mf in mask_files:
                parts = os.path.basename(mf).split('_')
                try:
                    idx = parts.index('mask.png')
                    obj_id = int(parts[idx - 1])
                    masks_dict[obj_id] = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                except (ValueError, IndexError):
                    continue

            rgb_img   = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            coord, color, segment, instance, ray = get_point_cloud_from_rgbd(
                rgb_img, depth_img, masks_dict, K, R_w2c, t_w2c, depth_scale
            )
            if len(coord) == 0:
                continue

            # 障碍物合并回背景（开放词汇任务里未标注点统一为 0）
            segment[segment == 41]   = 0
            instance[instance == 41] = 0

            # 该帧实际存在的 obj_id（非背景）
            raw_obj_ids = sorted(int(x) for x in np.unique(segment) if x != 0)
            if not raw_obj_ids:
                continue   # 没有标注物体，跳过

            # 重映射为连续 1-base 整数
            seg_remapped = np.zeros_like(segment, dtype=np.int32)
            for new_id, raw_id in enumerate(raw_obj_ids, start=1):
                seg_remapped[segment == raw_id] = new_id

            # labels[i] 对应 seg_remapped == i+1
            labels = [OBJ_ID_TO_TEXT.get(obj_id, f"object_{obj_id}") for obj_id in raw_obj_ids]

            # ── 保存 ──────────────────────────────────────────────
            os.makedirs(sample_dir, exist_ok=True)
            np.save(os.path.join(sample_dir, 'coord.npy'),   coord.astype(np.float32))
            np.save(os.path.join(sample_dir, 'color.npy'),   color.astype(np.uint8))
            np.save(os.path.join(sample_dir, 'segment.npy'), seg_remapped.astype(np.int32))
            np.save(os.path.join(sample_dir, 'normal.npy'),  ray.astype(np.float32))
            with open(os.path.join(sample_dir, 'labels.json'), 'w') as f:
                json.dump(labels, f)

            # ── PLY 可视化（debug 下精细，正常下稀疏）──────────────
            vis_dir = os.path.join(out_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            step = 1 if debug else 2

            save_ply_manual(coord[::step], color[::step],
                            os.path.join(vis_dir, f"{name_stem}_rgb.ply"))
            save_ply_manual(coord[::step], generate_semantic_colors(seg_remapped[::step]),
                            os.path.join(vis_dir, f"{name_stem}_seg.ply"))

            if debug:
                print(f"\n[DEBUG] {name_stem}")
                print(f"  raw obj_ids : {raw_obj_ids}")
                print(f"  labels      : {labels}")
                print(f"  seg unique  : {np.unique(seg_remapped).tolist()}")
                print(f"  N points    : {len(coord)}")
                print(f"  saved to    : {sample_dir}")
                sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",          required=True,
                        help="datasets/bopask-data/handal-qa-train BopAsk 数据目录，含 images/depth_maps/masks 子目录")
    parser.add_argument("--out_dir",          required=True,
                        help="Pointcept/data/concerto/bopask-openvocab 输出目录（Pointcept data/concerto/bopask）")
    parser.add_argument("--bop_original_dir", required=True,
                        help="datasets/bop_original/handal BOP 原始数据目录，含 val/scene_XXXXXX/scene_camera.json")
    parser.add_argument("--split_type",       default="val",
                        choices=["val", "test", "train"],
                        help="scene_camera.json 所在的 split 子目录名")
    parser.add_argument("--debug",            action="store_true",
                        help="只处理第一帧，保存 PLY 后退出")
    args = parser.parse_args()
    parse_bopask(args.src_dir, args.out_dir, args.bop_original_dir, args.split_type, args.debug)