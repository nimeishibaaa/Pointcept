import os
import json
import glob
from collections.abc import Sequence
from .defaults import DefaultDataset
from .builder import DATASETS

@DATASETS.register_module()
class HandalDataset(DefaultDataset):
    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            split_file = os.path.join(self.data_root, "splits", f"{split}.json")
            if os.path.isfile(split_file):
                with open(split_file) as f:
                    split_data = json.load(f)
                    if isinstance(split_data, dict):
                        for key, val in split_data.items():
                            data_list.append(os.path.join(self.data_root, val["pointclouds"]))
                    elif isinstance(split_data, list):
                        data_list.extend([os.path.join(self.data_root, data) for data in split_data])
            else:
                # fallback to DefaultDataset logic
                if os.path.isfile(os.path.join(self.data_root, split)):
                    with open(os.path.join(self.data_root, split)) as f:
                        data_list += [os.path.join(self.data_root, data) for data in json.load(f)]
                else:
                    data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

    def get_data_name(self, idx):
        # The directory name will be scene_xxxxxx_frame_xxxxxx
        return os.path.basename(self.data_list[idx % len(self.data_list)])
