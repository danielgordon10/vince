import copy
from typing import Dict, Tuple, Optional

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from torch import nn

import constants
from models.base_model import BaseModel
from utils import loss_util
from utils import util_functions
from utils.util_functions import to_uint8


class VinceModel(BaseModel):
    def __init__(self, args):
        super(VinceModel, self).__init__(args)
        self.args = args
        self.num_frames = self.args.num_frames

        # Network stuff
        self.feature_extractor = self.args.backbone(self.args, -2)
        resnet_output_channels = self.feature_extractor.output_channels
        self.output_channels = resnet_output_channels

        if self.args.use_attention:
            self.average_layers = pt_util.AttentionPool2D(resnet_output_channels, keepdim=False, return_masks=True)
        else:
            self.average_layers = nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), pt_util.RemoveDim((2, 3)))

        self.feature_extractor = pt_util.get_data_parallel(self.feature_extractor, args.feature_extractor_gpu_ids)
        self.feature_extractor_device = args.feature_extractor_gpu_ids[0]

        self.embedding = nn.Sequential(
            nn.Linear(self.output_channels, self.output_channels),
            constants.NONLINEARITY(),
            nn.Linear(self.output_channels, self.args.vince_embedding_size),
        )
        if self.args.jigsaw:
            self.jigsaw_linear = nn.Linear(self.output_channels, self.output_channels)
            self.jigsaw_embedding = nn.Sequential(
                nn.Linear(self.output_channels * 9, self.output_channels),
                constants.NONLINEARITY(),
                nn.Linear(self.output_channels, self.args.vince_embedding_size),
            )
        if self.args.inter_batch_comparison:
            if self.num_frames > 1:
                diag_mask = pt_util.from_numpy(
                    scipy.linalg.block_diag(
                        *[np.ones((self.num_frames, self.num_frames), dtype=np.bool)]
                         * (self.args.batch_size // self.num_frames)
                    )
                ).to(device=self.device)
                self.similarity_mask = torch.cat(
                    (
                        diag_mask,
                        torch.zeros(
                            (self.args.batch_size, self.args.vince_queue_size), device=self.device, dtype=torch.bool
                        ),
                    ),
                    dim=1,
                )

            eye = torch.eye(self.args.batch_size, device=self.device, dtype=torch.bool)
            self.eye_mask = torch.cat(
                (
                    eye,
                    torch.zeros(
                        (self.args.batch_size, self.args.vince_queue_size), device=self.device, dtype=torch.bool
                    ),
                ),
                dim=1,
            )

        if self.args.use_imagenet:
            self.imagenet_decoders = nn.ModuleList(
                [
                    nn.Linear(self.output_channels, 1000),
                    nn.Sequential(
                        nn.Linear(self.output_channels, self.output_channels),
                        constants.NONLINEARITY(),
                        nn.Linear(self.output_channels, 1000),
                    ),
                ]
            )
            self.num_imagenet_decoders = len(self.imagenet_decoders)

    def to(self, device):
        super(VinceModel, self).to(device)
        self.feature_extractor.to(self.feature_extractor_device)

    def vince_parameters(self):
        params = (
                list(self.feature_extractor.parameters())
                + list(self.embedding.parameters())
                + list(self.average_layers.parameters())
        )
        if self.args.jigsaw:
            params += list(self.jigsaw_linear.parameters()) + list(self.jigsaw_embedding.parameters())
        return params

    @staticmethod
    def split_dict_by_type(batch_types, batch_sizes, dict_to_split):
        # Splits a batch into sub-batches based on the batch_type and batch_size
        num_total = 0
        mini_batch_list = []
        assert "queue_vectors" not in dict_to_split
        for ind, (batch_type, batch_size) in enumerate(zip(batch_types, batch_sizes)):
            mini_batch = {
                key: (val[ind] if len(val) == len(batch_types) else val[num_total: num_total + batch_size])
                for key, val in dict_to_split.items()
            }
            mini_batch["batch_type"] = batch_type
            mini_batch.pop("batch_types", None)  # delete key if exists
            mini_batch_list.append(mini_batch)
            num_total += batch_size
        return mini_batch_list

    def extract_features(self, inputs, run_average_layer=True):
        return_val = {}
        spatial_features = self.feature_extractor(inputs)
        return_val["spatial_features"] = spatial_features
        if run_average_layer:
            features = self.average_layers(spatial_features)
            if isinstance(features, tuple):
                features, attention_masks = features
                return_val["attention_masks"] = attention_masks
            return_val["extracted_features"] = features
        return return_val

    def get_embeddings(self, inputs, jigsaw=False, shuffle=False):
        data = inputs["data"]
        if shuffle:
            # Shuffle
            shuffle_order = torch.randperm(data.shape[0], device=self.device)
            unshuffle_order = torch.zeros(data.shape[0], dtype=torch.int64, device=self.device)
            unshuffle_order.index_copy_(0, shuffle_order, torch.arange(data.shape[0], device=self.device))
            data = data[shuffle_order].contiguous()

        if jigsaw:
            if (data.shape[2] % 3) != 0 or (data.shape[3] % 3) != 0:
                data = F.pad(data, (0, 3 - data.shape[3] % 3, 0, 3 - data.shape[2] % 3))
            # [N, C, H, W]
            data = pt_util.split_dim(data, 2, 3, data.shape[2] // 3)
            # [N, C, 3, H/3, W]
            data = pt_util.split_dim(data, 4, 3, data.shape[4] // 3)
            # [N, C, 3, H/3, 3, W/3]
            data = data.permute(0, 2, 4, 1, 3, 5).contiguous()
            # [N, 3, 3, C, H/3, W/3]
            data = pt_util.remove_dim(data, (1, 2))
            # [N*9, C, H/3, W/3]

        images = data.to(self.feature_extractor_device)
        return_val = self.extract_features(images)
        features = return_val["extracted_features"]

        if jigsaw:
            features = features.to(self.device)
            features = self.jigsaw_linear(features)
            features = pt_util.split_dim(features, 0, -1, 9)
            # Shuffle all permutations independently
            rand_orders = torch.stack([torch.randperm(9, device=features.device) for _ in range(features.shape[0])])
            features = features[
                pt_util.expand_new_dim(torch.arange(features.shape[0], device=features.device), 1, 9), rand_orders
            ]
            features = pt_util.remove_dim(features, 2)
            features = self.jigsaw_embedding(features)
            return_val["extracted_features"] = features
            output = features
        else:
            features = features.to(self.device)
            return_val["extracted_features"] = features
            output = self.embedding(features)

        return_val["prenorm_features"] = output
        output = F.normalize(output, dim=1)

        return_val["embeddings"] = output

        if shuffle:
            # Unshuffle
            return_val_new = {}
            for key, val in return_val.items():
                if isinstance(val, torch.Tensor):
                    val = val.to(self.device)
                    val = val[unshuffle_order]
                return_val_new[key] = val
            return_val = return_val_new

        if "batch_types" in inputs:
            return_val = self.split_dict_by_type(inputs["batch_types"], inputs["batch_sizes"], return_val)
        return return_val

    def forward(self, inputs: Dict[str, torch.Tensor]):
        return_val = copy.copy(inputs)
        features = return_val["extracted_features"]
        if inputs["data_source"] == "IN":
            imagenet_features = features.clone().detach()

        output = return_val["embeddings"]
        if "queue_embeddings" in inputs and "vince_similarities" not in inputs:
            queue_embeddings = inputs["queue_embeddings"]
            if self.args.inter_batch_comparison:
                if inputs["num_frames"] > 1:
                    mask = self.similarity_mask
                else:
                    mask = self.eye_mask

                if self.args.self_batch_comparison:
                    self_similarities = torch.mm(output, output.t())
                    return_val.update(
                        dict(
                            vince_self_similarities_mask=mask[
                                                         : self_similarities.shape[0], : self_similarities.shape[1]
                                                         ],
                            vince_self_similarities=self_similarities,
                        )
                    )
                negs = torch.cat((queue_embeddings, inputs["queue_vectors"]), dim=0)
                l_neg = torch.mm(output, negs.t())
                similarities = l_neg
            else:
                l_pos = torch.bmm(output[:, np.newaxis, :], queue_embeddings[:, :, np.newaxis]).squeeze(2)
                negs = inputs["queue_vectors"]
                l_neg = torch.mm(output, negs.t())
                similarities = torch.cat([l_pos, l_neg], dim=1)
                return_val["vince_l_pos"] = l_pos
                mask = torch.zeros(similarities.shape, dtype=torch.bool, device=similarities.device)
                mask[:, 0] = True

            # If batch is smaller, end elements should be left off
            return_val.update(
                dict(
                    vince_l_neg=l_neg,
                    vince_similarities=similarities,
                    vince_similarities_mask=mask[: similarities.shape[0], : similarities.shape[1]],
                )
            )

        if inputs["data_source"] == "IN":
            imagenet_features = imagenet_features[: inputs["imagenet_labels"].shape[0]]
            for ii, imagenet_decoder in enumerate(self.imagenet_decoders):
                output = imagenet_decoder(imagenet_features)
                return_val["imagenet_decoder_%d" % ii] = output

        return return_val

    def loss(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[Tuple[float, torch.Tensor]]]:
        if network_outputs is None:
            losses = {"nce_loss": None}
            if self.args.self_batch_comparison:
                losses["nce_loss_self"] = None

            if hasattr(self, "num_imagenet_decoders"):
                for ii in range(self.num_imagenet_decoders):
                    losses["imagenet_loss_%d" % ii] = None
            return losses

        losses = {}
        temperature = self.args.vince_temperature

        if "vince_similarities" in network_outputs:
            similarities = network_outputs["vince_similarities"]
            batch_size = similarities.shape[0]
            mask = network_outputs["vince_similarities_mask"]
            similarity_losses = loss_util.similarity_cross_entropy(similarities, temperature, batch_size, 1, mask)
            network_outputs.update({"vince_loss_" + key: val for key, val in similarity_losses.items()})
            losses["nce_loss"] = (1.0, similarity_losses["dist"])

            if self.args.self_batch_comparison:
                mask = network_outputs["vince_self_similarities_mask"]
                similarities = network_outputs["vince_self_similarities"]
                temperature = self.args.vince_self_temperature
                similarity_losses = loss_util.similarity_cross_entropy(similarities, temperature, batch_size, 1, mask)
                network_outputs.update({"vince_loss_self_" + key: val for key, val in similarity_losses.items()})
                losses["nce_loss_self"] = (1.0, similarity_losses["dist"])

        if network_outputs["data_source"] == "IN":
            imagenet_losses = [
                F.cross_entropy(network_outputs["imagenet_decoder_%d" % ii], network_outputs["imagenet_labels"])
                for ii in range(self.num_imagenet_decoders)
            ]
            for ii in range(self.num_imagenet_decoders):
                losses["imagenet_loss_%d" % ii] = (1.0, imagenet_losses[ii])

        return losses

    def get_metrics(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            metrics = {}
            if network_outputs is None:
                metrics.update(
                    {
                        "nce_accuracy_mean": None,
                        "nce_softmax_weight_mean": None,
                        "cosine_sim": None,
                        "cosine_sim_neg_max": None,
                    }
                )
                if self.args.self_batch_comparison:
                    metrics.update(
                        {"nce_accuracy_self_mean": None, "nce_softmax_weight_self_mean": None, "cosine_self_sim": None}
                    )

                if hasattr(self, "num_imagenet_decoders"):
                    for ii in range(self.num_imagenet_decoders):
                        metrics["imagenet_accuracy_%d" % ii] = None
                return metrics

            for key in ["", "self_"]:
                if "vince_" + key + "similarities" in network_outputs:
                    similarities = network_outputs["vince_" + key + "similarities"]
                    batch_size = similarities.shape[0]
                    mask = network_outputs["vince_" + key + "similarities_mask"]
                    mask_row_sum = mask.sum(-1)
                    use_float = mask_row_sum.min() != mask_row_sum.max()

                    if use_float:
                        float_mask = mask.float()
                        pos_sim = similarities * float_mask + -2 ** 20 * (1 - float_mask)
                        neg_sim = similarities * (1 - float_mask) + -2 ** 20 * float_mask
                    else:
                        pos_sim = similarities[mask].view(batch_size, -1)
                        neg_sim = similarities[~mask].view(batch_size, -1)

                    neg_sim_max = torch.max(neg_sim, dim=1, keepdim=True)[0]
                    nce_accuracy_mean = (pos_sim > neg_sim_max).to(torch.float32).mean()
                    nce_softmax_weight = network_outputs["vince_loss_" + key + "softmax_weight"]
                    pos_sim_mean = pos_sim.mean()
                    metrics.update(
                        {
                            "nce_accuracy_" + key + "mean": nce_accuracy_mean,
                            "nce_softmax_weight_" + key + "mean": nce_softmax_weight,
                            "cosine_" + key + "sim": pos_sim_mean,
                        }
                    )
                    if key is "":
                        metrics["cosine_sim_neg_max"] = neg_sim_max.mean()

            if network_outputs["data_source"] == "IN":
                for ii in range(self.num_imagenet_decoders):
                    predictions = torch.argmax(network_outputs["imagenet_decoder_%d" % ii], dim=1)
                    acc = (predictions == network_outputs["imagenet_labels"]).to(torch.float32).mean()
                    metrics["imagenet_accuracy_%d" % ii] = acc
            return metrics

    def get_image_output(self, network_outputs) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            image_output = {}

            # matching image
            batch_size, _, im_height, im_width = network_outputs["data"].shape

            inputs = network_outputs["data"]
            queue_inputs = network_outputs["queue_data"]
            inputs = to_uint8(inputs, padding=10)
            queue_inputs = to_uint8(queue_inputs, padding=10)
            num_frames = 1 if self.num_frames is None else self.num_frames
            inputs = pt_util.split_dim(inputs, 0, -1, num_frames)
            queue_inputs = pt_util.split_dim(queue_inputs, 0, -1, num_frames)
            images = []
            color = (255, 128, 0)
            for bb in range(min(len(inputs), max(2 * num_frames, int(32 / num_frames)))):
                for ss in range(num_frames):
                    image = inputs[bb, ss]
                    images.append(image)
                for ss in range(num_frames):
                    image = queue_inputs[bb, ss].copy()
                    image[:10, :, :] = color
                    image[-10:, :, :] = color
                    image[:, :10, :] = color
                    image[:, -10:, :] = color
                    images.append(image)

            n_cols = max(2 * num_frames, 8)
            n_rows = len(images) // n_cols
            subplot = drawing.subplot(images, n_rows, n_cols, im_width, im_height)
            image_output["images/inputs"] = subplot

            if "vince_similarities" in network_outputs:
                # Nearest neighbor image
                inputs = network_outputs["data"]
                queue_inputs = network_outputs["queue_data"]

                inputs = to_uint8(inputs, padding=10)
                queue_inputs = to_uint8(queue_inputs, padding=10)

                vince_similarities = network_outputs["vince_similarities"]
                logits = vince_similarities / self.args.vince_temperature
                vince_softmax = F.softmax(logits, dim=1)

                queue_images = network_outputs["queue_images"]

                n_neighbors = 9
                topk_val, topk_ind = torch.topk(vince_softmax, n_neighbors, dim=1, largest=True, sorted=True)
                topk_ind = pt_util.to_numpy(topk_ind)
                topk_val = pt_util.to_numpy(topk_val)

                label = network_outputs["vince_similarities_mask"]

                images = []
                rand_order = np.random.choice(batch_size, min(batch_size, n_neighbors + 1), replace=False)
                for bb in rand_order:
                    query_image = inputs[bb].copy()
                    color = (90, 46, 158)
                    if network_outputs["batch_type"] == "images":
                        # Different colors for imagenet vs videos.
                        color = (24, 178, 24)
                    query_image[:10, :, :] = color
                    query_image[-10:, :, :] = color
                    query_image[:, :10, :] = color
                    query_image[:, -10:, :] = color
                    images.append(query_image)
                    found_neighbor = False
                    for nn, neighbor in enumerate(topk_ind[bb]):
                        color = (128, 128, 128)
                        score = topk_val[bb, nn]

                        if self.args.inter_batch_comparison:
                            if neighbor < batch_size:
                                image = queue_inputs[neighbor].copy()
                                data_source = network_outputs["data_source"]
                            else:
                                # Offset by batch_size for the inter-batch negatives
                                offset = batch_size
                                image = to_uint8(queue_images[neighbor - offset], padding=10)
                                data_source = network_outputs["queue_data_sources"][neighbor - offset]
                        else:
                            if neighbor == 0:
                                image = queue_inputs[bb].copy()
                                data_source = network_outputs["data_source"]
                            else:
                                # Offset by 1 for the positive examples
                                image = to_uint8(queue_images[neighbor - 1], padding=10)
                                data_source = network_outputs["queue_data_sources"][neighbor - 1]

                        if label[bb, neighbor]:
                            if self.args.inter_batch_comparison and neighbor < batch_size:
                                found_neighbor = True
                                color = (255, 128, 0)
                            elif neighbor == 0:
                                found_neighbor = True
                                color = (255, 128, 0)
                            elif data_source == "self":
                                color = (144, 72, 0)
                            else:
                                color = (0, 0, 203)
                        elif data_source == "self":
                            color = (255, 0, 193)
                        if not found_neighbor and nn == n_neighbors - 1:
                            # Last one in row, couldn't match proper, put in just to show what it looks like.
                            image = queue_inputs[bb].copy()
                            color = (255, 0, 0)

                        if color == (128, 128, 128):
                            color = (90, 46, 158)
                            if data_source == "IN":
                                # Different colors for imagenet vs videos.
                                color = (24, 178, 24)
                        image[:10, :, :] = color
                        image[-10:, :, :] = color
                        image[:, :10, :] = color
                        image[:, -10:, :] = color
                        images.append(image)

                n_rows = n_neighbors + 1
                n_cols = n_neighbors + 1
                subplot = drawing.subplot(images, n_rows, n_cols, im_width, im_height)
                image_output["images/outputs"] = subplot

            if network_outputs["data_source"] == "IN":
                # imagenet image
                predictions = torch.argmax(network_outputs["imagenet_decoder_0"], dim=1)
                labels = network_outputs["imagenet_labels"]
                acc = pt_util.to_numpy(predictions == labels)
                batch_size = acc.shape[0]

                inputs = network_outputs["data"][:batch_size]
                inputs = to_uint8(inputs, padding=10)

                images = []
                rand_order = np.random.choice(len(inputs), min(len(inputs), 25), replace=False)
                scale_factor = im_width / 320.0

                for bb in rand_order:
                    correct = acc[bb]
                    image = inputs[bb].copy()
                    pred_cls = util_functions.imagenet_label_to_class(predictions[bb])
                    gt_cls = util_functions.imagenet_label_to_class(labels[bb])
                    if correct:
                        cls_str = pred_cls
                    else:
                        cls_str = "Pred: %s Actual %s" % (pred_cls, gt_cls)

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
                image_output["images/imagenet_outputs"] = subplot

            if "attention_masks" in network_outputs:
                # Attention image
                inputs = network_outputs["data"]
                inputs = to_uint8(inputs, padding=10)

                queue_inputs = network_outputs["queue_data"]
                queue_inputs = to_uint8(queue_inputs, padding=10)

                attention_masks = network_outputs["attention_masks"]
                attention_masks = pt_util.to_numpy(
                    F.interpolate(attention_masks, (im_height, im_width), mode="bilinear", align_corners=False).permute(
                        0, 2, 3, 1
                    )
                )
                attention_masks = np.pad(attention_masks, ((0, 0), (10, 10), (10, 10), (0, 0)), "constant")

                queue_attention_masks = network_outputs["queue_attention_masks"]
                queue_attention_masks = pt_util.to_numpy(
                    F.interpolate(
                        queue_attention_masks, (im_height, im_width), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1)
                )
                queue_attention_masks = np.pad(queue_attention_masks, ((0, 0), (10, 10), (10, 10), (0, 0)), "constant")

                rand_order = np.random.choice(len(inputs), min(len(inputs), 25), replace=False)

                subplots = []
                attention_color = np.array([255, 0, 0], dtype=np.float32)
                for bb in rand_order:
                    images = []
                    for img_src, mask_src in ((inputs, attention_masks), (queue_inputs, queue_attention_masks)):
                        image = img_src[bb].copy()
                        attention_mask = mask_src[bb].copy()
                        attention_mask -= attention_mask.min()
                        attention_mask /= attention_mask.max() + 1e-8
                        output = (attention_mask * attention_color) + (1 - attention_mask) * image
                        output = output.astype(np.uint8)
                        images.append(image)
                        images.append(output)
                    subplot = drawing.subplot(images, 2, 2, im_width, im_height)
                    subplots.append(subplot)

                n_cols = int(np.sqrt(len(subplots)))
                n_rows = len(subplots) // n_cols

                subplot = drawing.subplot(subplots, n_rows, n_cols, im_width * 2, im_height * 2, border=5)
                image_output["images/attention"] = subplot

        return image_output


class VinceQueueModel(BaseModel):
    def __init__(self, args, encoder: VinceModel):
        super(VinceQueueModel, self).__init__(args)
        self.queue_network = copy.deepcopy(encoder)
        self.vince_momentum = self.args.vince_momentum

        for param in self.queue_network.parameters():
            param.requires_grad = False

    def to(self, device):
        super(VinceQueueModel, self).to(device)
        self.queue_network.to(device)
        self._device = device

    def param_update(self, encoder_model: VinceModel, momentum: float):
        with torch.no_grad():
            source_params = self.queue_network.vince_parameters()
            target_params = encoder_model.vince_parameters()
            for queue_param, param in zip(source_params, target_params):
                queue_param.mul_(momentum).add_(1 - momentum, param.data.to(device=queue_param.device))

    def vince_update(self, encoder_model):
        self.param_update(encoder_model, self.vince_momentum)

    def forward(self, inputs, jigsaw=False, shuffle=True):
        with torch.no_grad():
            queue_data = inputs["queue_data"]
            output_mini_batches = self.queue_network.get_embeddings(
                {"data": queue_data, "batch_types": inputs["batch_types"], "batch_sizes": inputs["batch_sizes"]},
                jigsaw=jigsaw,
                shuffle=shuffle,
            )
            return_vals = []
            for outputs in output_mini_batches:
                return_val = {}
                for key, val in outputs.items():
                    if isinstance(val, torch.Tensor):
                        val = val.detach()  # detach just to be extra careful
                    return_val["queue_" + key] = val
                return_vals.append(return_val)
            return return_vals
