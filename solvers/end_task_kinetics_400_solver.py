from typing import Dict

import torch
from dg_util.python_utils import pytorch_util as pt_util

from models.end_task_kinetics_400_model import Kinetics400Model
from solvers.end_task_classifier_solver import EndTaskClassifierSolver
from utils.util_functions import kinetics_400_label_to_class


class EndTaskKinetics400Solver(EndTaskClassifierSolver):
    @property
    def ind_to_label_func(self):
        return kinetics_400_label_to_class

    def make_decoder_network(self, args):
        model = Kinetics400Model(args, self.feature_extractor.output_channels, 512, self.ind_to_label_func)
        print(model)
        return model

    def setup_model_param_groups(self):
        param_groups = []
        wd = 0 if self.freeze_feature_extractor else 1e-4
        param_group = {
            "params": self.model.parameters(),
            "lr": self.args.base_lr,
            "weight_decay": wd,
            "initial_lr": self.args.base_lr,
        }
        param_groups.append(param_group)
        return param_groups

    def setup_other(self):
        pass

    def convert_batch(self, batch, batch_type: str = "train") -> Dict:
        batch["data_source"] = "Kinetics400"
        data = batch["data"]
        batch_size, seq_len = data.shape[:2]
        data = pt_util.remove_dim(data, 1)
        batch["data"] = data
        batch["batch_type"] = ("images", len(batch["data"]))
        batch["batch_size"] = batch_size
        batch["num_frames"] = seq_len
        return super(EndTaskKinetics400Solver, self).convert_batch(batch)

    def forward(self, batch):
        if self.freeze_feature_extractor:
            with torch.no_grad():
                feature_extractor_outputs = self.feature_extractor.extract_features(batch["data"])
                extracted_features = feature_extractor_outputs["extracted_features"].detach()
        else:
            feature_extractor_outputs = self.feature_extractor.extract_features(batch["data"])
            extracted_features = feature_extractor_outputs["extracted_features"]
        extracted_features = extracted_features.to(self.model.device)
        extracted_features = pt_util.split_dim(extracted_features, 0, batch["batch_size"], batch["num_frames"])
        output = self.model(extracted_features)
        output = {"outputs": output}
        output.update(batch)
        return output
