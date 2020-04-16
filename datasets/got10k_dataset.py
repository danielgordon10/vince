import numpy as np
import torchvision.transforms as transforms
from dg_util.python_utils import pytorch_util as pt_util
from got10k.datasets import GOT10k

from siamfc_pytorch.pair_dataset import PairDataset
from siamfc_pytorch.siamfc_transforms import SiamFCTransforms


def fliplr(x):
    # Copy because needs to be contiguous with positive stride
    return np.fliplr(x).copy()


class GOT10kDataset(PairDataset):
    def __init__(self, args, data_subset="train"):
        seqs = GOT10k(args.data_path, subset=data_subset, return_meta=True)
        self.cfg = args.cfg
        pair_transform = SiamFCTransforms(
            exemplar_sz=self.cfg["exemplar_sz"], instance_sz=self.cfg["instance_sz"], context=self.cfg["context"]
        )
        if data_subset == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomApply([transforms.Lambda(fliplr)], 0.5),
                    pt_util.ToTensor(scale=255),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    pt_util.ToTensor(scale=255),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        super(GOT10kDataset, self).__init__(args, seqs, data_subset, pair_transform, transform)
