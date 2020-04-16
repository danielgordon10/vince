import abc
import numbers

import torch
import torchvision.transforms as transforms
from PIL import Image
from dg_util.python_utils import pytorch_util as pt_util

from utils import util_functions

__all__ = [
    "BasicImagenetTransform",
    "StandardVideoTransform",
    "SimCLRTransform",
    "JigsawTransform",
    "SunSceneTransform",
    "Kinetics400Transform",
    "GOT10KTransform",
    "RepeatedImagenetTransform",
    "MoCoV1ImagenetTransform",
    "MoCoV2ImagenetTransform",
]


class BaseTransform(abc.ABC):
    def __init__(self, size, data_subset="train"):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size
        self.data_subset = data_subset
        self._train_transform = None
        self._val_transform = None

    def make_train_transform(self):
        raise NotImplementedError

    def make_val_transform(self):
        raise NotImplementedError

    @property
    def train_transform(self):
        if self._train_transform is None:
            self._train_transform = self.make_train_transform()
        return self._train_transform

    @property
    def val_transform(self):
        if self._val_transform is None:
            self._val_transform = self.make_val_transform()
        return self._val_transform

    def __call__(self, inputs):
        if self.data_subset == "train":
            outputs = self.train_transform(inputs)
        elif self.data_subset == "val":
            outputs = self.val_transform(inputs)
        else:
            raise NotImplementedError("No transform for data_subset %s" % self.data_subset)
        return outputs


class BasicImagenetTransform(BaseTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1), ratio=(0.7, 1.4), interpolation=Image.BILINEAR),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def make_val_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.Resize((int(self.size[0] / 0.875), int(self.size[1] / 0.875)), interpolation=Image.BILINEAR),
                transforms.CenterCrop(self.size),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class StandardVideoTransform(BasicImagenetTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class SimCLRTransform(StandardVideoTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomApply([util_functions.RandomGaussianBlur(self.size[0] // 10)], p=0.5),
            ]
        )


class JigsawTransform(StandardVideoTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.7, 1.0)),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomApply([util_functions.RandomGaussianBlur(self.size[0] // 10)], p=0.5),
            ]
        )


class SunSceneTransform(BasicImagenetTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.7, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class Kinetics400Transform(BasicImagenetTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.5, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class GOT10KTransform(BasicImagenetTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                pt_util.ToPILImage(),
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class RepeatedImagenetTransform(BasicImagenetTransform):
    def __init__(self, size, data_subset="train", repeats=1, stack=True):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size
        self.repeats = repeats
        self.data_subset = data_subset
        self.stack = stack
        super(RepeatedImagenetTransform, self).__init__(size, data_subset)

    def __call__(self, image):
        if self.data_subset == "val":
            # For val have the first one be val transformed and rest be train transformed.
            outputs = [self.val_transform(image)]
            for rr in range(self.repeats - 1):
                outputs.append(self.train_transform(image))
            if self.stack:
                outputs = torch.stack(outputs, dim=0)
            return outputs
        else:
            outputs = []
            for rr in range(self.repeats):
                outputs.append(self.train_transform(image))
        if self.stack:
            outputs = torch.stack(outputs, dim=0)
        return outputs


class MoCoV1ImagenetTransform(RepeatedImagenetTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size, scale=(0.08, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                # transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class MoCoV2ImagenetTransform(RepeatedImagenetTransform):
    def make_train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.size, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                pt_util.ToTensor(scale=255),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomApply([util_functions.RandomGaussianBlur(self.size[0] // 10)], p=0.5),
            ]
        )
