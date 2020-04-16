import copy
import numbers

import cv2
import numpy as np
from dg_util.python_utils import bb_util
from dg_util.python_utils import image_util

__all__ = ["SiamFCTransforms"]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomStretch(object):
    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, img):
        interp = cv2.INTER_LINEAR
        """
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        """
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        out_size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
        return cv2.resize(img, out_size, interpolation=interp)


class RandomStretchBox(object):
    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, box):
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        box[4] = round(box[4] * scale)
        box[5] = round(box[5] * scale)
        return box


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.0)
        j = round((w - tw) / 2.0)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            img = cv2.copyMakeBorder(img, npad, npad, npad, npad, cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad

        return img[i: i + th, j: j + tw]


class CenterCropBox(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, box):
        box[2] = self.size[1] * box[2] / box[4]
        box[3] = self.size[0] * box[3] / box[5]
        box[4] = self.size[1]
        box[5] = self.size[0]
        return box


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i: i + th, j: j + tw]


class RandomCropBox(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, box):
        box[0] += (np.random.random() - 0.5) * (box[2] - self.size[1] / box[4] * box[2])
        box[1] += (np.random.random() - 0.5) * (box[3] - self.size[0] / box[5] * box[3])
        return box


class SiamFCTransforms(object):
    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transforms_z = Compose(
            [RandomStretch(), CenterCrop(instance_sz - 8), RandomCrop(instance_sz - 2 * 8), CenterCrop(exemplar_sz)]
        )
        self.box_transforms_z = Compose(
            [
                RandomStretchBox(),
                CenterCropBox(instance_sz - 8),
                RandomCropBox(instance_sz - 2 * 8),
                CenterCropBox(exemplar_sz),
            ]
        )
        self.transforms_x = Compose([RandomStretch(), CenterCrop(instance_sz - 8), RandomCrop(instance_sz - 2 * 8)])
        self.box_transforms_x = Compose(
            [RandomStretchBox(), CenterCropBox(instance_sz - 8), RandomCropBox(instance_sz - 2 * 8)]
        )

    def __call__(self, inputs):
        z, x, box_z, box_x = inputs

        z = self._crop_and_stretch(z, box_z, self.box_transforms_z)
        x = self._crop_and_stretch(x, box_x, self.box_transforms_x)
        return z, x

    def _crop_and_stretch(self, img, box, box_transforms):
        # Faster version of their crop and stretch functions which only computes the output image once instead of many
        # times.
        box_start = copy.deepcopy(box)
        box = self._get_crop_box(box, self.instance_sz)
        box = box_transforms(box)
        box[2:4] = np.maximum(box[2:4], 2)
        if np.any(np.array(box[2:4]) < 2):
            print("box is very small", box_start, box)
        xyxy = bb_util.xywh_to_xyxy(box[:4])
        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        img = image_util.get_cropped_input(img, xyxy, 1, box[4], cv2.INTER_LINEAR, avg_color)[0]
        return img

    def _get_crop_box(self, box, out_size):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 + (box[2] - 1) / 2, box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz
        box = [center[1], center[0], size, size, out_size, out_size]
        return box
