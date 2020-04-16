import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dg_util.python_utils import pytorch_util as pt_util

import constants

IMAGENET_CLASS_NAMES = json.load(
    open(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "info_files", "imagenet_class_names.json"))
)
SUN_SCENE_CLASS_NAMES = [
    line.strip()
    for line in open(
        os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "info_files", "sun_scene_class_names.txt")
    )
]
KINETICS_400_CLASS_NAMES = [
    line.strip()
    for line in open(
        os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "info_files", "kinetics_400_class_names.txt")
    )
]

YT8M_CLASS_NAMES = [
    line.strip()
    for line in open(
        os.path.join(os.path.dirname(__file__), os.pardir, "datasets", "info_files", "yt8m_class_names.txt")
    )
]


def to_uint8(images, padding=0):
    images = pt_util.to_numpy(images)
    if len(images.shape) == 4:
        images = images.transpose(0, 2, 3, 1)
        if padding > 0:
            images = np.pad(images, ((0, 0), (padding, padding), (padding, padding), (0, 0)), "constant")
    else:
        images = images.transpose(1, 2, 0)
        if padding > 0:
            images = np.pad(images, ((padding, padding), (padding, padding), (0, 0)), "constant")
    images *= constants.IMAGENET_STD
    images += constants.IMAGENET_MEAN
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images


class RandomCropOpenCV(object):
    def __init__(self, size, scale=(0.8, 1.0), ratio=0.5, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.output_ratio = float(self.size[1]) / self.size[0]
        self.scale = scale
        self.ratio = ratio
        self.ratio_range = 1 - self.ratio
        self.scale_range = self.scale[1] - self.scale[0]
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray):
        im_height, im_width = img.shape[:2]
        # print('initial', im_width, im_height)
        rand_ratio = np.random.random() * self.ratio_range + self.ratio
        rand_scale = np.random.random() * self.scale_range + self.scale[0]
        # print('ratio', rand_ratio)
        # print('scale', rand_scale)

        crop_size = min(im_width, im_height) * rand_scale
        crop_width = crop_size
        crop_height = crop_size

        if np.random.random() < 0.5:
            crop_height = crop_width * rand_ratio
        else:
            crop_width = crop_height * rand_ratio

        # print('initial crop size', crop_width, crop_height)

        if crop_width > im_width:
            crop_height *= im_width / crop_width
            crop_width = im_width - 1
            # print('w new crop size', crop_width, crop_height)
        if crop_height > im_height:
            crop_width *= im_height / crop_height
            crop_height = im_height - 1
            # print('h new crop size', crop_width, crop_height)

        crop_width = int(crop_width)
        crop_height = int(crop_height)
        # print('end crop size', crop_width, crop_height)

        start_x = np.random.randint(0, im_width - crop_width)
        start_y = np.random.randint(0, im_height - crop_height)
        crop = img[start_y: start_y + crop_height, start_x: start_x + crop_width]
        # print('crop', crop.shape)
        crop = cv2.resize(crop, self.size)
        # print('result crop', crop.shape)
        # print('\n')
        return crop


class RandomGaussianBlur(object):
    def __init__(self, kernel_size, sigma_range=(0.1, 2.0)):
        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.sigma_range = sigma_range
        self.kernel_range = (self.kernel_size - 1) * 0.5 - torch.arange(self.kernel_size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            rand_sigma = np.random.random() * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
            kernel = torch.exp(-0.5 / (rand_sigma ** 2) * (self.kernel_range ** 2))
            kernel /= max(1e-10, kernel.sum())

            added_dim = False
            if len(img.shape) != 4:
                added_dim = True
                img = img[np.newaxis, ...]
            n_channels = img.shape[1]
            kernel = kernel[np.newaxis, :].expand(n_channels, self.kernel_size)
            img = F.conv2d(
                img, kernel[:, np.newaxis, :, np.newaxis], padding=(self.kernel_size // 2, 0), groups=n_channels
            )
            img = F.conv2d(
                img, kernel[:, np.newaxis, np.newaxis, :], padding=(0, self.kernel_size // 2), groups=n_channels
            )
            if added_dim:
                img = img.squeeze(0)
            return img


def imagenet_label_to_class(ind):
    return IMAGENET_CLASS_NAMES[str(int(ind))][1]


def sun_scene_label_to_class(ind):
    return SUN_SCENE_CLASS_NAMES[int(ind)]


def kinetics_400_label_to_class(ind):
    return KINETICS_400_CLASS_NAMES[int(ind)]


def yt8m_label_to_class(ind):
    return YT8M_CLASS_NAMES[int(ind)]
