import glob
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import tqdm

from datasets.base_dataset import BatchConcatDataset
from datasets.base_multi_frame_dataset import VideoDataset
from utils import transforms

SAMPLE_NAME = "AA/AA2pFq9pFTA_000001.jpg"
LEN_SAMPLE_NAME = len(SAMPLE_NAME)
LEN_VID_NAME = len("AA2pFq9pFTA")
LEN_NUM_NAME = len("000001")
LEN_CLIP_NAME = len("0001")


class R2V2Dataset(VideoDataset, BatchConcatDataset):
    @staticmethod
    def get_video_name(name):
        return name[-LEN_NUM_NAME - LEN_VID_NAME - 5: -LEN_NUM_NAME - 5]

    @staticmethod
    def get_frame_id(name):
        return int(name[-LEN_NUM_NAME - 4: -4])

    def get_image_paths(self):
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_split_path, "*/*.jpg")))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key[:2], key + "_%06d.jpg" % ind)

    def __init__(
            self,
            args,
            data_subset: str = "train",
            transform=None,
            num_images_to_return=-1,
            shared_transform=False,
            repeatable=False,
            check_for_new_data=False,
    ):
        self.worker_id = -1

        size = (args.input_height, args.input_width)
        if transform is None:
            transform = transforms.StandardVideoTransform(size, data_subset)
        VideoDataset.__init__(self, args, data_subset, transform, num_images_to_return, check_for_new_data)
        self.repeatable = repeatable
        self.shared_transform = shared_transform

    def __getitem__(self, idx) -> Optional[Dict[str, torch.Tensor]]:
        path_key, frame_ids = self.path_info[idx]
        if self.repeatable:
            initial_seed = self.worker_id + int(idx)
        else:
            initial_seed = random.randint(0, 2 ** 31)
        self.set_rng(initial_seed)

        if not self.args.multi_frame:
            frame_ids = np.random.choice(frame_ids, 1)

        image_cache = {}
        images = []
        queue_images = []

        for _ in range(self.num_images_to_return):
            image_path_inds = np.random.choice(frame_ids, 2, replace=True)
            for ii, ind in enumerate(image_path_inds):
                path = self.get_image_name(path_key, ind)
                if path not in image_cache:
                    image = self.read_image(path)
                    if image is None:
                        return None
                    image_cache[path] = image
                image = image_cache[path]
                if self.shared_transform:
                    self.set_rng(initial_seed)
                    image = self.transform(image)
                else:
                    image = self.transform(image)
                if ii == 0:
                    images.append(image)
                else:
                    queue_images.append(image)
        if len(images) < self.num_images_to_return:
            return None

        return {
            "data": images,
            "queue_data": queue_images,
            "labels": torch.ones(1),
            "ind": idx,
            "id": path_key,
            "keys_to_concat": ["data", "queue_data"],
        }


class GOT10KR2V2Dataset(R2V2Dataset):
    @staticmethod
    def get_video_name(name):
        return name.split(os.sep)[-2]

    def get_image_paths(self):
        return list(tqdm.tqdm(glob.iglob(os.path.join(self.data_split_path, "*/*.jpg"))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key, "%08d.jpg" % (ind + 1))
