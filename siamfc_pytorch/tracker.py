import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from got10k.trackers import Tracker

import constants
from siamfc_pytorch import ops


class SiamNet(pt_util.BaseModel):
    def __init__(self, backbone, head):
        super(SiamNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def feature_extractor(self, data):
        track_features = self.backbone.extract_features(data)
        return track_features["spatial_features"]

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):
    def __init__(self, name, args, cfg, net):
        super(TrackerSiamFC, self).__init__(name, True)
        self.args = args
        self.cfg = cfg
        self.net = net
        self.mean = pt_util.from_numpy(constants.IMAGENET_MEAN.copy())
        self.std = pt_util.from_numpy(constants.IMAGENET_STD.copy())
        self.visualize = False

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 + (box[2] - 1) / 2, box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg["response_up"] * self.cfg["response_sz"]
        self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg["scale_step"] ** np.linspace(
            -(self.cfg["scale_num"] // 2), self.cfg["scale_num"] // 2, self.cfg["scale_num"]
        )

        # exemplar and search sizes
        context = self.cfg["context"] * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * self.cfg["instance_sz"] / self.cfg["exemplar_sz"]

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz, out_size=self.cfg["exemplar_sz"], border_value=self.avg_color
        )

        # exemplar features
        if self.visualize:
            cv2.imshow("template", z[:, :, ::-1])
            cv2.waitKey(1)
        z = self.image_to_torch(z)
        self.kernel = self.net.extract_features(z)

    def make_mean_and_std(self, inputs):
        _, mean = pt_util.fix_broadcast(inputs, self.mean)
        _, std = pt_util.fix_broadcast(inputs, self.std)
        mean = mean.to(dtype=inputs.dtype, device=inputs.device)
        std = std.to(dtype=inputs.dtype, device=inputs.device)
        return mean, std

    def input_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs = inputs.to(torch.float32)
            mean, std = self.make_mean_and_std(inputs)
            # Do first one not in place to make sure it's not overwriting the original.
            inputs = inputs - mean
            inputs.div_(std)
            return inputs

    def inverse_input_transform(self, inputs: torch.Tensor, uint8=False) -> torch.Tensor:
        with torch.no_grad():
            mean, std = self.make_mean_and_std(inputs)
            # Do first one not in place to make sure it's not overwriting the original.
            inputs = inputs * std
            inputs.add_(mean)
            inputs.clamp_(0, 255)
            if uint8:
                inputs = inputs.to(torch.uint8)
            return inputs

    def image_to_torch(self, img):
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        img = pt_util.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.net.device)
        return self.input_transform(img)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [
            ops.crop_and_resize(
                img, self.center, self.x_sz * f, out_size=self.cfg["instance_sz"], border_value=self.avg_color
            )
            for f in self.scale_factors
        ]
        if self.visualize:
            search_images = drawing.subplot(x, 1, 3, x[0].shape[1], x[0].shape[0], 5)
            cv2.imshow("search_image", search_images[:, :, ::-1])
            cv2.waitKey(1)
        x = np.stack(x, axis=0)
        x = self.image_to_torch(x)

        # responses
        x = self.net.extract_features(x)
        responses = self.net.head(self.kernel, x)
        # responses = torch.sigmoid(responses)
        # responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = F.interpolate(
            responses, size=(self.upscale_sz, self.upscale_sz), mode="bicubic", align_corners=False
        )
        responses = pt_util.to_numpy(responses.squeeze(1))
        if self.visualize:
            response_image = drawing.subplot(-responses, 1, 3, responses.shape[2], responses.shape[1], 5)
            cv2.imshow("response image", response_image)

        responses[: self.cfg["scale_num"] // 2] *= self.cfg["scale_penalty"]
        responses[self.cfg["scale_num"] // 2 + 1:] *= self.cfg["scale_penalty"]

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg["window_influence"]) * response + self.cfg["window_influence"] * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        if self.visualize:
            loc_result = response == response.max()
            loc_result = loc_result.astype(np.uint8) * 255
            loc_result = cv2.dilate(loc_result, np.ones((9, 9), dtype=loc_result.dtype), iterations=1)
            loc_result = np.tile(loc_result[..., np.newaxis], (1, 1, 1, 3))
            cv2.imshow(
                "response max",
                drawing.subplot([-response, loc_result], 1, 2, loc_result.shape[2], loc_result.shape[1], 5),
            )

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * self.cfg["total_stride"] * 1.0 / self.cfg["response_up"]
        disp_in_image = disp_in_instance * self.x_sz * self.scale_factors[scale_id] / self.cfg["instance_sz"]
        # disp_in_image = disp_in_response * self.x_sz * self.scale_factors[scale_id] / self.upscale_sz
        self.center += disp_in_image
        if self.visualize:
            print(
                "loc",
                loc,
                "original center change",
                disp_in_response,
                "center change",
                disp_in_image,
                "new center",
                self.center,
            )

        # update target size
        scale = (1 - self.cfg["scale_lr"]) * 1.0 + self.cfg["scale_lr"] * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array(
            [
                self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
                self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
                self.target_sz[1],
                self.target_sz[0],
            ]
        )
        if self.visualize:
            cv2.waitKey(0)

        return box

    def track(self, img_files, box, visualize=False):
        self.visualize = visualize
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        im_read_time = 0

        for f, img_file in enumerate(tqdm.tqdm(img_files)):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
