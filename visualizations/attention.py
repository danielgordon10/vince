import cv2
import numpy as np
from dg_util.python_utils import misc_util
from dg_util.python_utils import pytorch_util as pt_util


def draw_attention(image, attention, height_channel=0, width_channel=1):
    dtype = image.dtype
    image = pt_util.to_numpy(image)
    attention = pt_util.to_numpy(attention)
    attention = np.clip(attention, 0, 1)
    im_width = image.shape[width_channel]
    im_height = image.shape[height_channel]
    attention = misc_util.resize(
        attention,
        (im_width, im_height),
        interpolation=cv2.INTER_LINEAR,
        height_channel=height_channel,
        width_channel=width_channel,
    )
    image, attention = pt_util.fix_broadcast(image, attention)
    image = (image * (1 - attention) + 255 * attention).astype(dtype)
    return image
