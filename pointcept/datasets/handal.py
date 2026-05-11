import os
from .defaults import DefaultDataset
from .builder import DATASETS

@DATASETS.register_module()
class HandalDataset(DefaultDataset):
    def get_data_name(self, idx):
        # The directory name will be scene_xxxxxx_frame_xxxxxx
        return os.path.basename(self.data_list[idx % len(self.data_list)])
