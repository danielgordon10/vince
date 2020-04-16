from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util

from models.end_task_base_model import EndTaskBaseModel
from utils.util_functions import to_uint8


class Kinetics400Model(EndTaskBaseModel):
    def __init__(self, args, num_input_channels, hidden_size, ind_to_label_func=None):
        super(Kinetics400Model, self).__init__(args)
        self.num_classes = self.args.end_task_classifier_num_classes
        self.hidden_size = hidden_size
        self.ind_to_label_func = ind_to_label_func
        self.number_of_layers = 1
        self.lstm = nn.LSTM(num_input_channels, self.hidden_size, batch_first=True, num_layers=self.number_of_layers)
        self.action_unembed = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, features):
        output, (hidden, cell) = self.lstm(features)  # To see the whole sequence and embed it into a hidden vector
        last_hidden = hidden[-1]
        action_pred = self.action_unembed(last_hidden)
        return action_pred

    def loss(self, network_outputs):
        if network_outputs is None:
            return {"kinetics_400_action_loss": None}
        loss = F.cross_entropy(network_outputs["outputs"], network_outputs["labels"])
        return {"kinetics_400_action_loss": (1.0, loss)}

    def get_metrics(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            if network_outputs is None:
                return {"accuracy": None}

            predictions = torch.argmax(network_outputs["outputs"], dim=1)
            accuracy = (predictions == network_outputs["labels"]).to(torch.float32).mean().item()
            return {"accuracy": accuracy}

    def get_image_output(self, network_outputs):
        with torch.no_grad():
            image_output = {}
            predictions = torch.argmax(network_outputs["outputs"], dim=1)
            labels = network_outputs["labels"]

            batch_size = network_outputs["batch_size"]
            seq_len = network_outputs["num_frames"]

            acc = pt_util.to_numpy(predictions == labels)

            inputs = network_outputs["data"]
            inputs = to_uint8(inputs)
            im_height, im_width = inputs.shape[1:3]

            inputs = pt_util.split_dim(inputs, 0, batch_size, seq_len)

            rand_order = np.random.choice(len(inputs), min(len(inputs), seq_len), replace=False)

            scale_factor = im_width / 320.0
            images = []
            for bb in rand_order:
                correct = acc[bb]
                image_seq = inputs[bb].copy()
                pred_cls = self.ind_to_label_func(predictions[bb])
                gt_cls = self.ind_to_label_func(labels[bb])
                for ii, image in enumerate(image_seq):
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
                    if ii == 0:
                        image = drawing.draw_contrast_text_cv2(
                            image, "P: " + pred_cls, (10, 10 + int(30 * scale_factor))
                        )
                        if not correct:
                            image = drawing.draw_contrast_text_cv2(
                                image, "GT: " + gt_cls, (10, 10 + int(2 * 30 * scale_factor))
                            )
                    images.append(image)

            n_cols = seq_len
            n_rows = len(images) // n_cols

            subplot = drawing.subplot(images, n_rows, n_cols, im_width, im_height)
            image_output["images/classifier_outputs"] = subplot
            return image_output
