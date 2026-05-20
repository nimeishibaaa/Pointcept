import os
import json
import argparse
import glob
import numpy as np
from collections import defaultdict


def generate_splits(dataset_path, val_ratio=0.1, seed=42):
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)

    # 只收录有 labels.json 的帧（即开放词汇有效帧）
    all_frames = glob.glob(os.path.join(dataset_path, "scene_*", "scene_*_frame_*"))
    valid_frames = [f for f in all_frames if os.path.isfile(os.path.join(f, "labels.json"))]

    if not valid_frames:
        print(f"No valid frames with labels.json found in {dataset_path}. Run preprocess_bopask.py first.")
        return

    print(f"Found {len(valid_frames)} frames with labels.json (out of {len(all_frames)} total).")

    # 按场景分组，防止数据泄漏
    scene_to_frames = defaultdict(list)
    for frame_path in valid_frames:
        scene_id = os.path.basename(os.path.dirname(frame_path))
        scene_to_frames[scene_id].append(frame_path)

    scenes = sorted(scene_to_frames.keys())
    print(f"Found {len(scenes)} scenes.")

    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)

    n_val = max(1, int(len(scenes) * val_ratio))
    val_scenes   = set(scenes[:n_val])
    train_scenes = set(scenes[n_val:])

    train_frames = [f for s in train_scenes for f in scene_to_frames[s]]
    val_frames   = [f for s in val_scenes   for f in scene_to_frames[s]]

    # 检查 val 是否包含 train 未见过的类别
    def collect_labels(frames):
        cats = set()
        for f in frames:
            with open(os.path.join(f, "labels.json")) as fp:
                cats.update(json.load(fp))
        return cats

    train_cats = collect_labels(train_frames)
    val_cats   = collect_labels(val_frames)
    unseen = val_cats - train_cats
    if unseen:
        print(f"[WARNING] Val has {len(unseen)} unseen categories not in train: {unseen}")

    def save_split(frames, name):
        rel_paths = [os.path.relpath(f, dataset_path) for f in sorted(frames)]
        out = {f"item_{i}": {"pointclouds": p} for i, p in enumerate(rel_paths)}
        with open(os.path.join(split_path, f"{name}.json"), "w") as fp:
            json.dump(out, fp, indent=2)
        scene_ids = {os.path.basename(os.path.dirname(f)) for f in frames}
        print(f"\n[{name}.json]  {len(frames)} frames across {len(scene_ids)} scenes")
        for sid in sorted(scene_ids):
            print(f"  {sid}: {len(scene_to_frames[sid])} frames")

    save_split(train_frames, "train")
    save_split(val_frames,   "val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                        help="Path to preprocessed bopask dataset")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of scenes to use for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_splits(args.dataset_root, val_ratio=args.val_ratio, seed=args.seed)