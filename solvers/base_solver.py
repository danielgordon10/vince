import abc
from typing import Dict

import numpy as np
import torch
import tqdm
from dg_util.python_utils import tensorboard_logger
from dg_util.python_utils.average_meter import RollingAverageMeter
from torch import nn
from torch import optim

try:
    from apex import amp, optimizers

    AMP = True
except ImportError:
    AMP = False


class BaseSolver(abc.ABC):
    def __init__(
            self, args, train_logger: tensorboard_logger.Logger = None, val_logger: tensorboard_logger.Logger = None
    ):
        self.args = args
        self.use_apex = AMP and self.args.use_apex
        self.input_size = args.input_size
        self.model: nn.Module = None
        self.logger_iteration = 0
        self.train_logger = None
        self.val_logger = None
        if not args.debug:
            self.train_logger = train_logger
            self.val_logger = val_logger
        self.time_meters = {}
        self.metric_meters = {}
        self.loss_meters = {}
        self.iteration = 0
        self.epoch = 0
        self.optimizer: optim.Optimizer = None
        self.freeze_feature_extractor = self.args.freeze_feature_extractor
        self.setup_dataloader()
        self.setup_other()
        self.setup_model()
        self.setup_optimizer()

    @property
    def device(self):
        torch_devices = self.args.pytorch_gpu_ids
        device = torch_devices[0]
        return device

    @property
    def model_name(self):
        if self.model is None:
            return "Unknown"
        return type(self.model).__name__

    @property
    def solver_name(self):
        return type(self).__name__

    @property
    def full_name(self):
        return self.solver_name + "_" + self.model_name

    def setup_dataloader(self):
        raise NotImplementedError

    @property
    def iterations_per_epoch(self):
        raise NotImplementedError

    def setup_other(self):
        # For setup after dataset stuff so they don't take crazy amounts of RAM.
        raise NotImplementedError

    def get_batch(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def setup_optimizer(self):
        raise NotImplementedError

    def adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        out_base_lr = self.args.base_lr
        for param_group in self.optimizer.param_groups:
            in_lr = param_group["initial_lr"]
            out_lr = in_lr
            if self.args.lr_decay_type == "cos":  # cosine lr schedule
                out_lr *= 0.5 * (1.0 + np.cos(np.pi * self.epoch / self.args.epochs))
            else:  # stepwise lr schedule
                for milestone in self.args.lr_step_schedule:
                    out_lr *= 0.1 if self.epoch >= milestone else 1.0
            param_group["lr"] = out_lr
            if in_lr == self.args.base_lr:
                out_base_lr = out_lr
        if self.train_logger is not None:
            self.train_logger.scalar_summary(
                "metrics/%s/epoch" % self.full_name, self.epoch, step=self.iteration, increment_counter=False
            )
            self.train_logger.scalar_summary(
                "metrics/%s/lr" % self.full_name, out_base_lr, step=self.iteration, increment_counter=False
            )
        print("Epoch", self.epoch, "Learning rate", out_base_lr)
        return out_base_lr

    def reset_epoch(self):
        self.logger_iteration = 0
        self.time_meters.update(
            dict(
                total_time=RollingAverageMeter(self.args.log_frequency),
                data_cache_time=RollingAverageMeter(self.args.log_frequency),
                forward_time=RollingAverageMeter(self.args.log_frequency),
                metrics_time=RollingAverageMeter(self.args.log_frequency),
                backward_time=RollingAverageMeter(self.args.log_frequency),
            )
        )
        self.metric_meters.update(
            {metric: RollingAverageMeter(self.args.log_frequency) for metric in self.model.get_metrics(None).keys()}
        )
        self.loss_meters.update(
            {key: RollingAverageMeter(self.args.log_frequency) for key in self.model.loss(None).keys()}
        )
        if len(self.loss_meters) > 1:
            self.loss_meters["total_loss"] = RollingAverageMeter(self.args.log_frequency)
        self.adjust_learning_rate()
        self.model.train()
        if self.train_logger is not None:
            self.train_logger.network_conv_summary(self.model, self.iteration)

    def run_train_iteration(self):
        raise NotImplementedError

    def run_n_train_iterations(self, num_iterations: int):
        self.reset_epoch()
        for _ in tqdm.tqdm(range(num_iterations)):
            self.run_train_iteration()

    def run_val(self):
        raise NotImplementedError

    def save(self, num_to_keep=-1):
        self.model.save(self.iteration, num_to_keep)
