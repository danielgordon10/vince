import os

import numpy
import numpy as np
import torch
from dg_util.python_utils import misc_util
from torch import nn

numpy.set_printoptions(precision=4)
torch.set_printoptions(precision=4, sci_mode=False)


def batch_norm_layer(channels):
    return nn.BatchNorm2d(channels)


def nonlinearity():
    return nn.ReLU(inplace=True)


NONLINEARITY = nonlinearity
NORM_LAYER = batch_norm_layer
TIME_STR = misc_util.get_time_str()
BASE_LOG_DIR = "logs"

CHECK_FOR_NEW_DATA = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
COOKIE_PATH = os.path.join(os.path.dirname(__file__), "youtube_scrape", "cookies.txt")
