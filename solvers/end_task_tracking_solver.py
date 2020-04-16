import os
import time
from typing import Dict

import torch
from got10k import experiments

from models.end_task_tracking_model import SiamFCTrackingModel
from siamfc_pytorch.tracker import SiamNet, TrackerSiamFC
from solvers.end_task_base_solver import EndTaskBaseSolver


def parse_args(**kwargs):
    # default parameters
    cfg = {
        # basic parameters
        "out_scale": 0.001,
        "exemplar_sz": 120,
        "instance_sz": 255,
        "context": 0.5,
        # inference parameters
        "scale_num": 3,
        "scale_step": 1.0375,
        "scale_lr": 0.59,
        "scale_penalty": 0.9745,
        "window_influence": 0.176,
        "response_sz": 17,
        "response_up": 16,
        "total_stride": 8,
        # train parameters
        "epoch_num": 50,
        "batch_size": 8,
        "num_workers": 32,
        "initial_lr": 1e-2,
        "ultimate_lr": 1e-5,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "r_pos": 16,
        "r_neg": 0,
    }

    for key, val in kwargs.items():
        if key in cfg:
            cfg.update({key: val})
    return cfg


class EndTaskTrackingSolver(EndTaskBaseSolver):
    def __init__(self, args, train_logger=None, val_logger=None):
        self.cfg = parse_args()
        args.cfg = self.cfg
        super(EndTaskTrackingSolver, self).__init__(args, train_logger, val_logger)

    def make_decoder_network(self, args) -> torch.nn.Module:
        return SiamFCTrackingModel(args, self.cfg, self.feature_extractor.output_channels)

    def setup_model_param_groups(self):
        wd = 0 if self.freeze_feature_extractor else 1e-4
        param_group = {
            "params": self.model.parameters(),
            "lr": self.args.base_lr,
            "weight_decay": wd,
            "initial_lr": self.args.base_lr,
        }
        return [param_group]

    def setup_other(self):
        pass

    def convert_batch(self, batch, batch_type: str = "train") -> Dict:
        exemplar_images, track_images = batch
        batch = dict(
            data=exemplar_images,
            track_data=track_images,
            data_source="GOT10k",
            batch_type=("images", len(track_images)),
        )
        return super(EndTaskTrackingSolver, self).convert_batch(batch)

    def forward(self, batch):
        if self.freeze_feature_extractor:
            with torch.no_grad():
                exemplar_features = self.feature_extractor.extract_features(batch["data"])
                track_features = self.feature_extractor.extract_features(batch["track_data"])
                exemplar_features = exemplar_features["spatial_features"].detach().to(self.model.device)
                track_features = track_features["spatial_features"].detach().to(self.model.device)
        else:
            exemplar_features = self.feature_extractor.extract_features(batch["data"])
            track_features = self.feature_extractor.extract_features(batch["track_data"])
            exemplar_features = exemplar_features["spatial_features"].to(self.model.device)
            track_features = track_features["spatial_features"].to(self.model.device)

        output = self.model(exemplar_features, track_features)
        output = {"responses": output}
        output.update(batch)
        return output

    def run_eval(self):
        with torch.no_grad():
            self.feature_extractor.eval()
            self.model.eval()
            tracker_net = SiamNet(self.feature_extractor, self.model)
            name = "SiamFC_" + self.model_name + "_" + self.args.description
            tracker = TrackerSiamFC(name, self.args, self.cfg, tracker_net)
            version = 2015
            experiment = experiments.ExperimentOTB(os.path.join(self.args.data_path, "otb100"))  # , version=version)
            # experiment = experiments.ExperimentVOT(os.path.join(self.args.data_path, "vot"), version=version, read_image=False)
            # experiment = experiments.ExperimentGOT10k(self.args.data_path)
            experiment.run(tracker, visualize=False)
            t_start = time.time()
            t_end = time.time()
            tracker_names = os.listdir("results/OTB%s" % str(version))
            print("time", t_end - t_start)
