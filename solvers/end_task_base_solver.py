import abc
import copy
import os
import time
import traceback
from typing import Dict, List

import torch
import tqdm
from dg_util.python_utils.average_meter import RollingAverageMeter, AverageMeter
from dg_util.python_utils.persistent_dataloader import PersistentDataLoader
from torch import nn

import constants
from models.vince_model import VinceModel
from solvers.base_solver import BaseSolver

try:
    from apex import amp, optimizers
except ImportError:
    pass


class EndTaskBaseSolver(BaseSolver, abc.ABC):
    def __init__(self, args, train_logger=None, val_logger=None):
        self.train_loader = None
        self.val_loader = None
        self.batch_iter = None
        self.feature_extractor: nn.Module = None

        super(EndTaskBaseSolver, self).__init__(args, train_logger, val_logger)
        self.train_batch_counter = 0

    def setup_dataloader(self):
        if not self.args.disable_dataloader:
            self.train_loader = PersistentDataLoader(
                dataset=None,
                num_workers=min(self.args.num_workers, 40),
                pin_memory=True,
                device=self.device,
                never_ending=True,
            )

            self.val_loader = PersistentDataLoader(
                dataset=None,
                num_workers=min(self.args.num_workers, 40),
                pin_memory=False,
                device=self.device,
                never_ending=True,
            )

            self.train_loader.set_dataset(
                self.args.dataset(self.args, "train"),
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=self.args.dataset.collate_fn,
                worker_init_fn=self.args.dataset.worker_init_fn,
            )
            self.batch_iter = iter(self.train_loader)
            self.val_loader.set_dataset(
                self.args.dataset(self.args, "val"),
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=self.args.dataset.collate_fn,
                worker_init_fn=self.args.dataset.worker_init_fn,
            )

    @property
    def iterations_per_epoch(self):
        return len(self.train_loader)

    def setup_model_param_groups(self) -> List[Dict]:
        raise NotImplementedError

    @staticmethod
    def create_optimizer(param_groups, base_lr):
        return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=1e-4)

    def setup_optimizer(self):
        base_lr = self.args.base_lr
        param_groups = self.setup_model_param_groups()

        if not self.freeze_feature_extractor:
            param_group = {
                "params": self.feature_extractor.parameters(),
                "lr": base_lr,
                "weight_decay": 1e-4,
                "initial_lr": base_lr,
            }
            param_groups.append(param_group)

        # optimizer = torch.optim.SGD(param_groups, lr=base_lr, weight_decay=1e-4, momentum=0.9)
        optimizer = self.create_optimizer(param_groups, base_lr)

        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = base_lr

        if self.use_apex:
            (self.feature_extractor, self.model), optimizer = amp.initialize([self.feature_extractor, self.model],
                                                                             optimizer, opt_level="O1",
                                                                             max_loss_scale=65536)

        print("optimizer", optimizer)
        self.optimizer = optimizer
        self.print_optimizer()

    @property
    def solver_model_name(self):
        return type(self).__name__[: -len("Solver")] + "Model"

    def setup_feature_extractor(self):
        args = copy.deepcopy(self.args)
        args.title = os.path.join(args.title, "VinceModel")
        args.checkpoint_dir = os.path.join(args.base_logdir, args.title, *(args.checkpoint_dir.split(os.sep)[2:]))
        args.long_save_checkpoint_dir = os.path.join(
            args.base_logdir, args.title, *(args.long_save_checkpoint_dir.split(os.sep)[2:-1]), constants.TIME_STR
        )
        args.tensorboard_dir = os.path.join(
            args.base_logdir, args.title, *(args.checkpoint_dir.split(os.sep)[2:-1]), constants.TIME_STR
        )
        self.feature_extractor = VinceModel(args)
        print(self.feature_extractor)
        self.feature_extractor.restore()
        self.feature_extractor.to(self.device)
        if self.freeze_feature_extractor:
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()

    def make_decoder_network(self, args) -> torch.nn.Module:
        raise NotImplementedError

    def setup_model(self):
        self.setup_feature_extractor()
        args = copy.deepcopy(self.args)
        args.title = os.path.join(args.title, self.solver_model_name)
        args.checkpoint_dir = os.path.join(args.base_logdir, args.title, *(args.checkpoint_dir.split(os.sep)[2:]))
        args.long_save_checkpoint_dir = os.path.join(
            args.base_logdir, args.title, *(args.long_save_checkpoint_dir.split(os.sep)[2:-1]), constants.TIME_STR
        )
        args.tensorboard_dir = os.path.join(
            args.base_logdir, args.title, *(args.checkpoint_dir.split(os.sep)[2:-1]), constants.TIME_STR
        )
        self.model = self.make_decoder_network(args)

        self.iteration = self.model.restore()
        self.model.to(self.device)

        if self.train_loader is not None:
            self.epoch = self.iteration // len(self.train_loader.dataset)
        if self.freeze_feature_extractor:
            if self.train_logger is not None:
                self.train_logger.network_conv_summary(self.feature_extractor.feature_extractor.module, self.iteration)

    def reset_epoch(self):
        super(EndTaskBaseSolver, self).reset_epoch()
        if not self.freeze_feature_extractor and self.train_logger is not None:
            self.train_logger.network_conv_summary(self.feature_extractor.feature_extractor.module, self.iteration)
        if self.freeze_feature_extractor:
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()
        self.model.train()

    def convert_batch(self, batch, batch_type: str = "train") -> Dict:
        batch = {
            key: (val.to(self.model.device, non_blocking=True) if isinstance(val, torch.Tensor) else val)
            for key, val in batch.items()
        }
        return batch

    def get_batch(self):
        iter_output = next(self.batch_iter)
        self.train_batch_counter += 1
        if self.train_batch_counter == len(self.train_loader):
            print("Hit stop iteration. End of epoch.")
            self.train_logger.scalar_summary(
                "metrics/%s/epoch" % self.full_name, self.epoch, step=self.iteration, increment_counter=False
            )
            self.train_logger.scalar_summary(
                "metrics/%s/lr" % self.full_name,
                self.optimizer.param_groups[0]["lr"],
                step=self.iteration,
                increment_counter=False,
            )
            self.train_batch_counter = 0
            raise StopIteration
        return self.convert_batch(iter_output, "train")

    def get_val_batch(self):
        # Useful for never_ending persistent dataloader which will never raise StopIteration on its own.
        val_iter = iter(self.val_loader)
        for _ in range(len(self.val_loader)):
            iter_output = next(val_iter)
            yield self.convert_batch(iter_output, "val")
        raise StopIteration

    def forward(self, batch):
        if self.freeze_feature_extractor:
            with torch.no_grad():
                features = self.feature_extractor.extract_features(batch["data"])
                extracted_features = features["extracted_features"].to(self.model.device).detach()
        else:
            features = self.feature_extractor.extract_features(batch["data"])
            extracted_features = features["extracted_features"].to(self.model.device)

        output = self.model(extracted_features)

        output.update(features)
        output.update(batch)
        return output

    def run_train_iteration(self):
        total_t_start = time.time()
        t_start = time.time()
        try:
            image_batch = self.get_batch()
        except StopIteration:
            return

        t_end = time.time()
        self.time_meters["data_cache_time"].update(t_end - t_start)
        t_start = time.time()

        output = self.forward(image_batch)
        loss_dict = self.model.loss(output)

        t_end = time.time()
        self.time_meters["forward_time"].update(t_end - t_start)
        t_start = time.time()

        metrics = self.model.get_metrics(output)

        total_loss = 0
        for key, val in loss_dict.items():
            weighted_loss = val[0] * val[1]
            total_loss = total_loss + weighted_loss
            self.loss_meters[key].update(weighted_loss)
        if "total_loss" in self.loss_meters:
            self.loss_meters["total_loss"].update(total_loss)
        loss = total_loss

        try:
            assert torch.isfinite(loss)
        except AssertionError as re:
            import pdb
            traceback.print_exc()
            pdb.set_trace()
            print("anomoly", re)
            raise re

        for key, val in metrics.items():
            self.metric_meters[key].update(val)

        t_end = time.time()
        self.time_meters["metrics_time"].update(t_end - t_start)

        t_start = time.time()

        self.optimizer.zero_grad()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        t_end = time.time()
        self.time_meters["backward_time"].update(t_end - t_start)

        if self.logger_iteration % self.args.image_log_frequency == 0:
            image_output = self.model.get_image_output(output)
            if self.train_logger is not None:
                for key, val in image_output.items():
                    if key.startswith("images"):
                        if isinstance(val, list):
                            for vv, item in enumerate(val):
                                self.train_logger.image_summary(
                                    self.full_name + "_" + key[len("images/"):], item, self.iteration + vv, False
                                )
                        else:
                            self.train_logger.image_summary(
                                self.full_name + "_" + key[len("images/"):], val, self.iteration, False
                            )

        if self.logger_iteration % self.args.log_frequency == 0:
            log_dict = {"times/%s/%s" % (self.full_name, key): val.val for key, val in self.time_meters.items()}
            log_dict.update({"losses/%s/%s" % (self.full_name, key): val.val for key, val in self.loss_meters.items()})
            log_dict.update(
                {"metrics/%s/%s" % (self.full_name, key): val.val for key, val in self.metric_meters.items()}
            )
            if self.train_logger is not None:
                self.train_logger.dict_log(log_dict, self.iteration)

        self.iteration += self.args.batch_size

        if self.logger_iteration % self.args.save_frequency == 0:
            self.save(5)

        total_t_end = time.time()
        self.time_meters["total_time"].update(total_t_end - total_t_start)
        self.logger_iteration += 1

    def run_val(self):
        with torch.no_grad():
            self.feature_extractor.eval()
            self.model.eval()
            time_meters = dict(
                total_time=RollingAverageMeter(self.args.log_frequency),
                data_cache_time=RollingAverageMeter(self.args.log_frequency),
                forward_time=RollingAverageMeter(self.args.log_frequency),
                metrics_time=RollingAverageMeter(self.args.log_frequency),
            )
            loss_meters = {key: RollingAverageMeter(self.args.log_frequency) for key in self.model.loss(None).keys()}
            if len(loss_meters) > 1:
                loss_meters["total_loss"] = RollingAverageMeter(self.args.log_frequency)
            metric_meters = {
                metric: RollingAverageMeter(self.args.log_frequency) for metric in self.model.get_metrics(None).keys()
            }

            epoch_loss_meters = {"epoch_" + key: AverageMeter() for key in loss_meters.keys()}
            epoch_metric_meters = {"epoch_" + key: AverageMeter() for key in metric_meters.keys()}

            step_on = self.iteration

            total_t_start = time.time()
            for ii, image_batch in enumerate(tqdm.tqdm(self.get_val_batch(), total=len(self.val_loader))):
                batch_size = image_batch["data"].shape[0]

                t_end = time.time()
                time_meters["data_cache_time"].update(t_end - total_t_start)
                t_start = time.time()
                output = self.forward(image_batch)
                loss_dict = self.model.loss(output)

                t_end = time.time()
                time_meters["forward_time"].update(t_end - t_start)
                t_start = time.time()

                metrics = self.model.get_metrics(output)
                if ii % self.args.image_log_frequency == 0:
                    image_output = self.model.get_image_output(output)

                total_loss = 0
                for key, val in loss_dict.items():
                    weighted_loss = val[0] * val[1]
                    total_loss = total_loss + weighted_loss
                    loss_meters[key].update(weighted_loss)
                    epoch_loss_meters["epoch_" + key].update(weighted_loss, batch_size)
                if "total_loss" in loss_meters:
                    loss_meters["total_loss"].update(total_loss)
                    epoch_loss_meters["epoch_total_loss"].update(total_loss, batch_size)
                loss = total_loss

                try:
                    assert torch.isfinite(loss)
                except AssertionError:
                    import pdb
                    pdb.set_trace()
                    print("Loss is infinite")

                for key, val in metrics.items():
                    metric_meters[key].update(val)
                    epoch_metric_meters["epoch_" + key].update(val, batch_size)

                t_end = time.time()
                time_meters["metrics_time"].update(t_end - t_start)

                if ii % self.args.image_log_frequency == 0:
                    if self.val_logger is not None:
                        for key, val in image_output.items():
                            if isinstance(val, list):
                                for vv, item in enumerate(val):
                                    self.val_logger.image_summary(self.full_name + "/" + key, item, step_on + vv, False)
                            else:
                                self.val_logger.image_summary(self.full_name + "/" + key, val, step_on, False)

                if ii % self.args.log_frequency == 0:
                    log_dict = {"times/%s/%s" % (self.full_name, key): val.val for key, val in time_meters.items()}
                    log_dict.update(
                        {"losses/%s/%s" % (self.full_name, key): val.val for key, val in loss_meters.items()}
                    )
                    log_dict.update(
                        {"metrics/%s/%s" % (self.full_name, key): val.val for key, val in metric_meters.items()}
                    )
                    if self.val_logger is not None:
                        self.val_logger.dict_log(log_dict, step_on)

                step_on += self.args.batch_size
                total_t_end = time.time()
                time_meters["total_time"].update(total_t_end - total_t_start)
                total_t_start = time.time()

        log_dict = {"epoch/losses/%s/%s" % (self.full_name, key): val.avg for key, val in epoch_loss_meters.items()}
        log_dict.update(
            {"epoch/metrics/%s/%s" % (self.full_name, key): val.avg for key, val in epoch_metric_meters.items()}
        )
        if self.val_logger is not None:
            self.val_logger.dict_log(log_dict, step_on)

    def run_eval(self):
        self.val_loader = PersistentDataLoader(
            dataset=None,
            num_workers=min(self.args.num_workers, 40),
            pin_memory=True,
            device=self.device,
            never_ending=True,
        )
        self.val_loader.set_dataset(
            self.args.dataset(self.args, "val"),
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.args.dataset.collate_fn,
            worker_init_fn=self.args.dataset.worker_init_fn,
        )
        self.run_val()

    def save(self, num_to_keep=-1):
        self.model.save(self.iteration, num_to_keep)
        if not self.freeze_feature_extractor:
            self.feature_extractor.save(self.iteration, num_to_keep)
