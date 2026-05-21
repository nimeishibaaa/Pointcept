"""
BopAsk Dataset

pointcept/datasets/bopask.py

两个类：
  BopaskDataset          —— 闭集语义/实例分割，兼容标准 PPT + SemSegEvaluator 管道
  BopaskOpenVocabDataset —— 开放词汇检索，输出 texts (List[str]) 和 segment [N, T] 二值 mask

期望的磁盘数据目录结构（每个 scene 一个子目录）：
  <scene_dir>/
    coord.npy      [N, 3]  float32
    color.npy      [N, 3]  float32   (RGB, 0-255 范围)
    normal.npy     [N, 3]  float32
    segment.npy    [N]     int32     0=背景, 1..K=实例 ID
    labels.json    List[str]         下标 i → segment==i 对应的文本标签
                   例如: ["background", "red spatula", "hammer"]

splits/ 目录下存放划分文件，支持两种格式：
  train.json  →  dict: {"scene_001": {"pointclouds": "scenes/scene_001"}, ...}
              或 list: ["scenes/scene_001", "scenes/scene_002", ...]
"""

import os
import json
import glob
import numpy as np
from copy import deepcopy
from collections.abc import Sequence

from .defaults import DefaultDataset
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


# ──────────────────────────────────────────────────────────────
# 共享工具
# ──────────────────────────────────────────────────────────────

def _build_data_list(data_root, split):
    """从 splits/<split>.json 构建数据路径列表，兼容 dict 和 list 两种格式。"""
    data_list = []
    split_file = os.path.join(data_root, "splits", f"{split}.json")
    if os.path.isfile(split_file):
        with open(split_file) as f:
            split_data = json.load(f)
        if isinstance(split_data, dict):
            for val in split_data.values():
                data_list.append(os.path.join(data_root, val["pointclouds"]))
        elif isinstance(split_data, list):
            data_list.extend([os.path.join(data_root, d) for d in split_data])
    elif os.path.isfile(os.path.join(data_root, split)):
        with open(os.path.join(data_root, split)) as f:
            data_list += [os.path.join(data_root, d) for d in json.load(f)]
    else:
        data_list += glob.glob(os.path.join(data_root, split, "*"))
    return data_list


# ──────────────────────────────────────────────────────────────
# 闭集数据集（标准管道）
# ──────────────────────────────────────────────────────────────

@DATASETS.register_module()
class BopaskDataset(DefaultDataset):
    """
    闭集版 BopAsk，兼容 SemSegEvaluator / PPT 微调管道。
    segment: [N] int32，整数实例/类别 ID（DefaultDataset 原生支持）。
    """

    def get_data_list(self):
        if isinstance(self.split, str):
            splits = [self.split]
        elif isinstance(self.split, Sequence):
            splits = list(self.split)
        else:
            raise NotImplementedError

        data_list = []
        for sp in splits:
            data_list.extend(_build_data_list(self.data_root, sp))
        return data_list

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])


# ──────────────────────────────────────────────────────────────
# 开放词汇数据集
# ──────────────────────────────────────────────────────────────

@DATASETS.register_module()
class BopaskOpenVocabDataset(BopaskDataset):
    """
    开放词汇版 BopAsk，供 OpenVocabPPT 使用。

    相比 BopaskDataset 的差异：
      - get_data() 加载 labels.json，把 segment [N] int → [N, T] float 二值 mask
      - prepare_test_data() 把 texts 注入每个 fragment，供推理时使用
      - 支持 pad_num_texts：把不足 T_max 的场景补零，保证同批 Num_texts 一致

    参数
    ----
    pad_num_texts : int | None
        将所有场景的 texts/segment 填充到固定长度 T_max。
        设 None 则不填充（要求同一 batch 内所有样本 Num_texts 相同）。
    background_label : str
        背景类的文本标签，用于验证 labels.json[0] 是否是背景。
        设 None 则不校验。
    """

    VALID_ASSETS = ["coord", "color", "normal", "segment"]

    def __init__(
        self,
        pad_num_texts=None,
        background_label=None,
        **kwargs,
    ):
        self.pad_num_texts = pad_num_texts
        self.background_label = background_label
        super().__init__(**kwargs)

    # ── 数据加载 ──────────────────────────────────────────────

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        # 加载 .npy 几何/颜色文件
        data_dict = {}
        for asset in os.listdir(data_path):
            if not asset.endswith(".npy"):
                continue
            key = asset[:-4]
            if key not in self.VALID_ASSETS:
                continue
            data_dict[key] = np.load(os.path.join(data_path, asset))

        data_dict["name"] = name
        data_dict["split"] = self.get_split_name(idx)

        # 类型标准化
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        # ── 加载文本标签 ──────────────────────────────────────
        labels_path = os.path.join(data_path, "labels.json")
        assert os.path.isfile(labels_path), (
            f"labels.json not found in {data_path}.\n"
            "Expected format: [\"background\", \"red spatula\", \"hammer\"]"
        )
        with open(labels_path) as f:
            texts = json.load(f)

        assert isinstance(texts, list) and len(texts) > 0, (
            f"labels.json must be a non-empty list of strings, got: {texts}"
        )
        if self.background_label is not None:
            assert texts[0] == self.background_label, (
                f"labels.json[0] should be '{self.background_label}', got '{texts[0]}'"
            )

        # ── 整数 segment [N] → 二值 mask [N, T] ──────────────
        seg_int = data_dict["segment"].reshape(-1).astype(np.int32)
        num_texts = len(texts)
        binary_mask = np.zeros((len(seg_int), num_texts), dtype=np.float32)
        # Note: raw object IDs are mapped to 1, 2, ... in preprocessing.
        # texts[0] corresponds to object 1 (seg_int == 1).
        for i in range(num_texts):
            binary_mask[:, i] = (seg_int == i + 1).astype(np.float32)
        
        # If a point is background (seg_int == 0), it will have all 0s in binary_mask.
        # We can explicitly set these to -1 if we want to ignore them.


        # ── 可选 padding 到固定 T_max ─────────────────────────
        if self.pad_num_texts is not None:
            T_max = self.pad_num_texts
            assert num_texts <= T_max, (
                f"Scene {name} has {num_texts} text queries, "
                f"exceeds pad_num_texts={T_max}"
            )
            if num_texts < T_max:
                pad_cols = T_max - num_texts
                binary_mask = np.concatenate(
                    [binary_mask, np.zeros((len(seg_int), pad_cols), dtype=np.float32)],
                    axis=1,
                )
                texts = texts + [""] * pad_cols

        data_dict["segment"] = binary_mask   # [N, T] float32
        data_dict["texts"] = texts           # List[str], len = T
        return data_dict

    # ── 训练模式 ──────────────────────────────────────────────

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    # ── 测试/验证模式 ─────────────────────────────────────────

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)

        # texts 和 segment 是场景级别的，从 data_dict 里提取保存
        result_dict = {
            "name":    data_dict.pop("name"),
            "texts":   data_dict.pop("texts"),     # List[str]，供 evaluator 使用
            "segment": data_dict.pop("segment"),   # [N_vox, T] after GridSample
        }
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict, \
                "origin_segment present but inverse missing after GridSample"
            result_dict["origin_segment"] = data_dict.pop("origin_segment")  # [N_orig, T]
            result_dict["inverse"] = data_dict.pop("inverse")

        # 数据增强 × 多 aug → fragment 列表
        data_dict_list = [aug(deepcopy(data_dict)) for aug in self.aug_transform]

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                data_part = self.test_crop(data_part) if self.test_crop else [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
            # texts 注入每个 fragment，让模型的 forward 能拿到
            fragment_list[i]["texts"] = result_dict["texts"]

        result_dict["fragment_list"] = fragment_list
        return result_dict