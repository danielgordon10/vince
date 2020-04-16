import abc
from typing import Dict

import torch

from models.linear_model import MultiLinearModel
from solvers.end_task_base_solver import EndTaskBaseSolver
from utils.transforms import BasicImagenetTransform
from utils.util_functions import imagenet_label_to_class
from utils.util_functions import sun_scene_label_to_class


class EndTaskClassifierSolver(EndTaskBaseSolver, abc.ABC):
    @property
    def ind_to_label_func(self):
        raise NotImplementedError

    def make_decoder_network(self, args):
        model = MultiLinearModel(
            args,
            self.feature_extractor.output_channels,
            self.args.end_task_classifier_num_classes,
            [1, 2],
            self.ind_to_label_func,
            dropout=0,
        )
        print(model)
        return model

    def setup_model_param_groups(self):
        param_groups = []
        wd = 0 if self.freeze_feature_extractor else 1e-4
        for mm, model in enumerate(self.model.classifiers):
            param_group = {
                "params": model.parameters(),
                "lr": self.args.base_lr * 0.01 ** mm,
                "weight_decay": wd,
                "initial_lr": self.args.base_lr * 0.01 ** mm,
            }
            param_groups.append(param_group)
        return param_groups


class EndTaskImagenetSolver(EndTaskClassifierSolver):
    def setup_other(self):
        pass

    @staticmethod
    def get_transform(size, data_subset="train"):
        return BasicImagenetTransform(size, data_subset)

    @property
    def ind_to_label_func(self):
        return imagenet_label_to_class

    @staticmethod
    def create_optimizer(param_groups, base_lr):
        return torch.optim.SGD(param_groups, lr=base_lr, weight_decay=0, momentum=0.9)

    def convert_batch(self, batch, batch_type: str = "train") -> Dict:
        data, labels = batch
        batch = {"data": data, "classifier_labels": labels, "data_source": "IN", "batch_type": ("images", len(data))}
        return super(EndTaskImagenetSolver, self).convert_batch(batch)


class EndTaskSunSceneSolver(EndTaskClassifierSolver):
    @property
    def ind_to_label_func(self):
        return sun_scene_label_to_class

    def setup_model_param_groups(self):
        param_groups = []
        wd = 0 if self.freeze_feature_extractor else 1e-4
        for mm, model in enumerate(self.model.classifiers):
            param_group = {
                "params": model.parameters(),
                "lr": self.args.base_lr,
                "weight_decay": wd,
                "initial_lr": self.args.base_lr,
            }
            param_groups.append(param_group)
        return param_groups

    def setup_other(self):
        pass

    def convert_batch(self, batch, batch_type: str = "train") -> Dict:
        batch["data_source"] = "SUN_Scenes"
        batch["batch_type"] = ("images", len(batch["data"]))
        return super(EndTaskSunSceneSolver, self).convert_batch(batch)
