from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dg_util.python_utils import bb_util
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util

from models.end_task_base_model import EndTaskBaseModel
from siamfc_pytorch.losses import BalancedLoss
from utils.util_functions import to_uint8


class SiamFCTrackingModel(EndTaskBaseModel):
    def __init__(self, args, cfg, input_channels, out_scale=0.001):
        super(SiamFCTrackingModel, self).__init__(args)
        self.out_scale = out_scale
        self.cfg = cfg
        self.criterion = BalancedLoss()

        self._labels = None
        self.search_patch_decoder = nn.Sequential(nn.Conv2d(input_channels, 256, 1))
        self.exemplar_decoder = nn.Sequential(nn.Conv2d(input_channels, 256, 1))
        self.upscale_sz = self.cfg["response_up"] * self.cfg["response_sz"]
        self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()
        # search scale factors
        self.scale_factors = self.cfg["scale_step"] ** np.linspace(
            -(self.cfg["scale_num"] // 2), self.cfg["scale_num"] // 2, self.cfg["scale_num"]
        )

        # exemplar and search sizes
        self.target_sz = 1
        self.context = self.cfg["context"] * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + self.context))
        self.x_sz = self.z_sz * self.cfg["instance_sz"] / self.cfg["exemplar_sz"]

    def forward(self, exemplar_features, features):
        exemplar_features = self.exemplar_decoder(exemplar_features)
        features = self.search_patch_decoder(features)
        output = self._fast_xcorr(exemplar_features, features) * self.out_scale
        return output

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

    def _create_labels(self, size):
        # skip if same sized labels already created
        if self._labels is not None:
            return self._labels[: size[0]]

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(
                dist <= r_pos, np.ones_like(x), np.where(dist < r_neg, np.ones_like(x) * 0.5, np.zeros_like(x))
            )
            # labels = np.exp(-0.5 * np.sqrt(x ** 2 + y ** 2))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg["r_pos"] / self.cfg["total_stride"]
        r_neg = self.cfg["r_neg"] / self.cfg["total_stride"]
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self._labels = torch.from_numpy(labels).to(self.device).float()

        return self._labels

    def loss(self, network_outputs=None):
        if network_outputs is None:
            return {"siam_tracking_loss": None}
        # calculate loss
        responses = network_outputs["responses"]
        labels = self._create_labels(responses.size())
        loss = self.criterion(responses, labels)
        # loss = F.binary_cross_entropy_with_logits(responses, labels)
        return {"siam_tracking_loss": (1.0, loss)}

    def get_metrics(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            if network_outputs is None:
                return {"dist": None, "center_dist": None, "mean_iou": None}
            metrics = {}
            responses = network_outputs["responses"]
            labels = self._create_labels(responses.size())
            responses_sigm = torch.sigmoid(responses)
            dist = torch.abs(responses_sigm - labels)
            metrics["dist"] = dist.mean()

            pred_boxes = self.prediction_to_box(responses)
            network_outputs["pred_boxes_cxcywh"] = pred_boxes

            gt_box_cxcywh = [0.5, 0.5, 0.5, 0.5]
            center_dist = torch.mean(torch.abs(pred_boxes[:2] - 0.5))
            metrics["center_dist"] = center_dist
            gt_box = bb_util.xywh_to_xyxy(gt_box_cxcywh)
            pred_boxes = bb_util.xywh_to_xyxy(pred_boxes)
            ious = bb_util.IOU_numpy(pred_boxes.T, gt_box)
            metrics["mean_iou"] = np.mean(ious)
            return metrics

    def prediction_to_box(self, responses):
        # returns cx cy w h
        with torch.no_grad():
            # locate target center
            loc = np.unravel_index(
                pt_util.to_numpy(torch.argmax(responses.view(responses.shape[0], -1), dim=-1)), responses.shape
            )
            loc = pt_util.from_numpy(loc).T
            loc = loc.to(torch.float32)
            loc[:, 2:] += 0.5  # treat as the center of the box, not the top left corner
            loc[:, 2:] = loc[:, 2:] / responses.shape[2]
            pred_box = torch.stack(
                [loc[:, 3], loc[:, 2], torch.full_like(loc[:, 0], 0.5), torch.full_like(loc[:, 0], 0.5)]
            )
            return pred_box

    def get_image_output(self, network_outputs) -> Dict[str, np.ndarray]:
        image_output = {}
        exemplar_images = to_uint8(network_outputs["data"])
        track_images = to_uint8(network_outputs["track_data"])
        responses = pt_util.to_numpy(network_outputs["responses"].squeeze(1))
        labels = pt_util.to_numpy(self._create_labels(responses.shape))
        batch_size, _, im_height, im_width = network_outputs["track_data"].shape

        images = []
        for exemplar_image, track_image, response, label in zip(exemplar_images, track_images, responses, labels):
            images.extend([exemplar_image, track_image, response, label])
            if len(images) > ((4 * 2) ** 2):
                break

        subplot = drawing.subplot(images, 4 * 2, 4 * 2, im_width, im_height)
        image_output["images/tracks"] = subplot

        return image_output
