from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from torch import nn

import constants
from models.base_model import BaseModel
from utils.util_functions import to_uint8


class MultiLayerLinearModel(nn.Module):
    def __init__(self, sizes, nonlinearity, dropout=0):
        super(MultiLayerLinearModel, self).__init__()
        network = []
        for ss in range(len(sizes) - 1):
            network.append(nn.Linear(sizes[ss], sizes[ss + 1]))
            if ss < len(sizes) - 2:
                network.append(nonlinearity())
                if dropout > 0:
                    network.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*network)

    def forward(self, inputs):
        return self.model(inputs)


class MultiLinearModel(BaseModel):
    def __init__(
            self,
            args,
            feature_size,
            num_classes,
            num_layers_for_each_model: List[int],
            ind_to_label_func=None,
            downsample_feature_size=None,
            dropout=0,
    ):
        super(MultiLinearModel, self).__init__(args)
        self.args = args
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.classifiers = []
        self.dropout = dropout
        for depth in num_layers_for_each_model:
            model = []
            feature_size = self.feature_size
            for ii in range(depth):
                if ii == depth - 1:
                    model.append(nn.Linear(feature_size, self.num_classes))
                else:
                    # Won't hit for single layer because it will hit the if
                    if ii == 0 and downsample_feature_size is not None:
                        feature_size = downsample_feature_size
                    model.append(nn.Linear(feature_size, feature_size))
                    model.append(nn.Dropout(dropout, inplace=True))
                    model.append(constants.NONLINEARITY())
            model = nn.Sequential(*model)
            self.classifiers.append(model)
        self.classifiers = nn.ModuleList(self.classifiers)
        self.num_classifiers = len(self.classifiers)
        self.ind_to_label_func = ind_to_label_func

    def forward(self, inputs):
        return_val = {}
        for ii in range(self.num_classifiers):
            return_val["classifier_output_%d" % ii] = self.classifiers[ii](inputs)
        return return_val

    def loss(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[Tuple[float, torch.Tensor]]]:
        losses = {}
        if network_outputs is None:
            for ii in range(self.num_classifiers):
                losses["classifier_loss_%d" % ii] = None
            return losses

        classifier_losses = [
            F.cross_entropy(network_outputs["classifier_output_%d" % ii], network_outputs["classifier_labels"])
            for ii in range(self.num_classifiers)
        ]
        for ii in range(self.num_classifiers):
            losses["classifier_loss_%d" % ii] = (1.0, classifier_losses[ii])
        return losses

    def get_metrics(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            metrics = {}
            if network_outputs is None:
                for ii in range(self.num_classifiers):
                    metrics["classifier_accuracy_%d" % ii] = None
                return metrics

            for ii in range(self.num_classifiers):
                predictions = torch.argmax(network_outputs["classifier_output_%d" % ii], dim=1)
                acc = (predictions == network_outputs["classifier_labels"]).to(torch.float32).mean()
                metrics["classifier_accuracy_%d" % ii] = acc
            return metrics

    def get_image_output(self, network_outputs):
        with torch.no_grad():
            image_output = {}
            predictions = torch.argmax(network_outputs["classifier_output_0"], dim=1)
            labels = network_outputs["classifier_labels"]
            acc = pt_util.to_numpy(predictions == labels)
            batch_size = acc.shape[0]

            if "attention_masks" in network_outputs:
                inputs = network_outputs["data"]
                im_height, im_width = inputs.shape[2:]
                inputs = to_uint8(inputs)

                attention_masks = network_outputs["attention_masks"]
                attention_masks = pt_util.to_numpy(
                    F.interpolate(attention_masks, (im_height, im_width), mode="bilinear", align_corners=False).permute(
                        0, 2, 3, 1
                    )
                )
                attention_masks = np.pad(attention_masks, ((0, 0), (10, 10), (10, 10), (0, 0)), "constant")

                rand_order = np.random.choice(len(inputs), min(len(inputs), 50), replace=False)

                images = []
                attention_color = np.array([255, 0, 0], dtype=np.float32)
                for bb in rand_order:
                    img_src = inputs
                    mask_src = attention_masks
                    image = img_src[bb].copy()
                    attention_mask = mask_src[bb].copy()
                    attention_mask -= attention_mask.min()
                    attention_mask /= attention_mask.max() + 1e-8
                    output = (attention_mask * attention_color) + (1 - attention_mask) * image
                    output = output.astype(np.uint8)
                    images.append(image)
                    images.append(output)

                n_cols = int(np.sqrt(len(images)))
                # if n_cols % 2 != 0:
                # n_cols += n_cols % 2
                n_rows = len(images) // n_cols

                subplot = drawing.subplot(images, n_rows, n_cols, im_width * 2, im_height * 2, border=5)
                image_output["images/attention"] = subplot

            images = []
            inputs = network_outputs["data"][:batch_size]
            inputs = to_uint8(inputs)
            im_height, im_width = inputs.shape[1:3]
            rand_order = np.random.choice(len(inputs), min(len(inputs), 25), replace=False)
            scale_factor = im_width / 320.0
            for bb in rand_order:
                correct = acc[bb]
                image = inputs[bb].copy()
                pred_cls = self.ind_to_label_func(predictions[bb])
                gt_cls = self.ind_to_label_func(labels[bb])
                if correct:
                    image[:10, :, :] = (0, 255, 0)
                    image[-10:, :, :] = (0, 255, 0)
                    image[:, :10, :] = (0, 255, 0)
                    image[:, -10:, :] = (0, 255, 0)
                else:
                    image[:10, :, :] = (255, 0, 0)
                    image[-10:, :, :] = (255, 0, 0)
                    image[:, :10, :] = (255, 0, 0)
                    image[:, -10:, :] = (255, 0, 0)
                image = drawing.draw_contrast_text_cv2(image, "P: " + pred_cls, (10, 10 + int(30 * scale_factor)))
                if not correct:
                    image = drawing.draw_contrast_text_cv2(
                        image, "GT: " + gt_cls, (10, 10 + int(2 * 30 * scale_factor))
                    )
                images.append(image)

            n_cols = int(np.sqrt(len(images)))
            n_rows = len(images) // n_cols

            subplot = drawing.subplot(images, n_rows, n_cols, im_width, im_height)
            image_output["images/classifier_outputs"] = subplot
            return image_output
