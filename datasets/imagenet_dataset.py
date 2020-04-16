import os

import torchvision.datasets as datasets

from datasets.base_dataset import BaseDataset
from utils.transforms import BasicImagenetTransform


class ImagenetDataset(datasets.ImageFolder, BaseDataset):
    def __init__(self, args, data_subset, transform=None, sample_inds=None):
        BaseDataset.__init__(self, args, data_subset)
        if transform is None:
            transform = BasicImagenetTransform(self.size, data_subset)
        datasets.ImageFolder.__init__(self, os.path.join(self.args.imagenet_data_path, data_subset), transform)
        if sample_inds is not None:
            self.samples = [self.samples[ii] for ii in sorted(sample_inds)]
