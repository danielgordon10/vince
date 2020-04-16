import abc
import functools
import operator
import os
import random
from typing import List, Dict, Union, Optional, Any

import numpy as np
import torch
from PIL import Image
from dg_util.python_utils import pytorch_util as pt_util
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, args, data_subset: str = "train"):
        self.args = args
        self.width = self.args.input_width
        self.height = self.args.input_height
        self.size = (self.height, self.width)
        self.data_subset = data_subset

    @property
    def name(self):
        return type(self).__name__

    def set_rng(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def worker_init_fn(worker_id):
        self = torch.utils.data.get_worker_info().dataset
        if isinstance(self, pt_util.IndexWrapperDataset):
            self = self.other_dataset
        self.worker_id = worker_id
        if hasattr(self, "repeatable") and self.repeatable:
            self.seed = worker_id
        else:
            self.seed = torch.initial_seed()
        if isinstance(self, BaseDataset):
            self.set_rng(self.seed % (2 ** 32 - 1))

    def read_image(self, path: str) -> Optional[Union[Any, np.ndarray]]:
        if not os.path.exists(path):
            print("Image is not found", path)
            return None
        try:
            image = Image.open(path)
            return image
        except:
            pass
            print("Could not read image", path)
            return None

    @staticmethod
    def collate_fn(batch):
        batch = default_collate(batch)
        return batch


class BatchConcatDataset(BaseDataset):
    @staticmethod
    # Special multi-dimensional concat
    def collate_fn(batch: List[Dict]) -> Dict:
        batch = list([bb for bb in batch if bb is not None])
        if len(batch) == 0:
            return {}
        assert isinstance(batch[0], dict), "only implemented for dicts"
        assert "keys_to_concat" in batch[0]

        keys_to_concat = batch[0]["keys_to_concat"]
        concat_items = {key: [bb[key] for bb in batch] for key in keys_to_concat}
        for bb in batch:
            for key in keys_to_concat:
                del bb[key]
            del bb["keys_to_concat"]
        batch = default_collate(batch)

        for key, items in concat_items.items():
            output_shape = [len(items)]
            while isinstance(items[0], list) or isinstance(items[0], tuple):
                output_shape.append(len(items[0]))
                items = functools.reduce(operator.iconcat, items, [])

            elem = items[0]

            out = None
            if torch.utils.data.get_worker_info() is not None:
                # Taken from Pytorch source.
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in items])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            items = [pt_util.from_numpy(dd) for dd in items]
            out_data = torch.stack(items, 0, out=out)
            out_data = out_data.view(*output_shape, *out_data.shape[1:])
            batch[key] = out_data
        return batch
