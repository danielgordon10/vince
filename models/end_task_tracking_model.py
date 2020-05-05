from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dg_util.python_utils import bb_util
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util

from models.end_task_base_model import EndTaskBaseModel
from siamfc_pytorch import losses
from utils.util_functions import to_uint8


class SiamFCTrackingModel(EndTaskBaseModel):
    def __init__(self, args, cfg, input_channels, out_scale=0.001):
        super(SiamFCTrackingModel, self).__init__(args)
        self.out_scale = out_scale
        self.cfg = cfg
        # self.criterion = losses.BalancedLoss()
        self.criterion = losses.FocalLoss()

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

    def loss(self, network_outputs=None):
        if network_outputs is None:
            return {"siam_tracking_loss": None}
        # calculate loss
        responses = network_outputs["responses"]
        labels = network_outputs['labels']
        loss = self.criterion(responses, labels.float())
        # loss = F.binary_cross_entropy_with_logits(responses, labels)
        return {"siam_tracking_loss": (1.0, loss)}

    def get_metrics(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            if network_outputs is None:
                return {"dist": None, "center_dist": None, "mean_iou": None}
            metrics = {}
            responses = network_outputs["responses"]
            labels = network_outputs['labels']
            responses_sigm = torch.sigmoid(responses)
            dist = torch.abs(responses_sigm - labels.float())
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
        responses = pt_util.to_numpy(network_outputs["responses"].squeeze(1)).copy()
        responses -= responses.min(axis=(1, 2), keepdims=True)
        responses /= responses.max(axis=(1, 2), keepdims=True)
        labels = pt_util.to_numpy(network_outputs['labels'].squeeze(1).float())
        responses += labels / 10
        batch_size, _, im_height, im_width = network_outputs["track_data"].shape

        images = []
        for exemplar_image, track_image, response in zip(exemplar_images, track_images, responses):
            images.extend([exemplar_image, track_image, response])
            if len(images) > ((3 * 2) ** 2):
                break

        subplot = drawing.subplot(images, 3 * 2, 3 * 2, im_width, im_height)
        image_output["images/tracks"] = subplot

        return image_output
