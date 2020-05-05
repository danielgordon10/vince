import copy
import os
import random
import time
import traceback
from queue import Queue
from threading import Thread

import numpy as np
import scipy.stats
import torch
import torchvision.datasets as datasets
import tqdm
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from dg_util.python_utils.average_meter import RollingAverageMeter, AverageMeter
from dg_util.python_utils.persistent_dataloader import PersistentDataLoader
from sklearn.neighbors import KDTree
from torch import optim

import constants
from datasets.npz_dataset import NPZDataset
from models.vince_model import VinceModel, VinceQueueModel
from solvers.base_solver import BaseSolver
from utils.storage_queue import StorageQueue

try:
    from apex import amp, optimizers
except ImportError:
    pass


class VinceSolver(BaseSolver):
    def __init__(self, args, train_logger=None, val_logger=None):
        # Declare these before super call because super sets them up.
        self.num_frames = args.num_frames
        self.train_loaders = []
        self.train_batch_iterators = []
        self.train_loader_counts = []
        self.train_batch_fns = []
        self.train_data_names = []
        self.val_loaders = []
        self.val_batch_fns = []
        self.val_data_names = []
        self.vince_queue: StorageQueue = None
        self.queue_model: VinceQueueModel = None
        self.batch_count = 0
        self.batch_queue = Queue(2)
        self.prefetch_thread = None
        self.kill_thread = False
        self.cifar_dataset: NPZDataset = None
        self.drawn_this_epoch = False

        super(VinceSolver, self).__init__(args, train_logger, val_logger)

    def setup_dataloader(self):
        torch_devices = self.args.pytorch_gpu_ids
        device = torch_devices[0]

        # Can use multiple datasets at once.
        # Delayed start. Create data loaders to make the processes, then create the datasets so they don't overwhelm
        # shared memory.
        print("creating data processes")
        t_start = time.time()
        if self.args.use_imagenet:
            imagenet_train_loader = PersistentDataLoader(
                dataset=None,
                num_workers=min(self.args.num_workers, 40),
                pin_memory=True,
                device=device,
                never_ending=True,
            )
            batch_iterator = iter(imagenet_train_loader)
            batch_fn = self.get_imagenet_batch
            self.train_loaders.append(imagenet_train_loader)
            self.train_batch_iterators.append(batch_iterator)
            self.train_batch_fns.append(batch_fn)
            self.train_loader_counts.append(0)
            self.train_data_names.append("ImageNet")

            imagenet_val_loader = PersistentDataLoader(
                dataset=None, num_workers=min(self.args.num_workers, 10), pin_memory=False, device=device
            )
            self.val_loaders.append(imagenet_val_loader)
            self.val_batch_fns.append(self.process_imagenet_data)
            self.val_data_names.append("ImageNet")

        if not self.args.disable_dataloader and self.args.use_videos:
            video_train_loader = PersistentDataLoader(
                dataset=None,
                num_workers=min(self.args.num_workers, 40),
                pin_memory=True,
                device=device,
                never_ending=True,
            )
            batch_iterator = iter(video_train_loader)
            batch_fn = self.get_video_batch
            self.train_loaders.append(video_train_loader)
            self.train_batch_iterators.append(batch_iterator)
            self.train_batch_fns.append(batch_fn)
            self.train_loader_counts.append(0)
            self.train_data_names.append("R2V2")

            video_val_loader = PersistentDataLoader(
                dataset=None, num_workers=min(self.args.num_workers, 20), pin_memory=False, device=device
            )
            self.val_loaders.append(video_val_loader)
            self.val_batch_fns.append(self.process_video_data)
            self.val_data_names.append("R2V2")

        print("done in %.3f" % (time.time() - t_start))
        print("creating datasets")
        t_start = time.time()

        # Now create the actual datasets
        if not self.args.disable_dataloader and self.args.use_imagenet:
            imagenet_train_loader.set_dataset(
                datasets.ImageFolder(
                    os.path.join(self.args.imagenet_data_path, "train"),
                    transform=self.args.transform(self.input_size, "train", 2 * self.num_frames, stack=False),
                ),
                batch_size=self.args.batch_size // self.num_frames,
                shuffle=True,
            )
            print(
                "Loaded ImageNet train",
                len(imagenet_train_loader.dataset),
                "images",
                len(imagenet_train_loader),
                "batches",
            )
            imagenet_val_loader.set_dataset(
                datasets.ImageFolder(
                    os.path.join(self.args.imagenet_data_path, "val"),
                    transform=self.args.transform(self.input_size, "val", 2 * self.num_frames, stack=False),
                ),
                batch_size=self.args.batch_size // self.num_frames,
                shuffle=True,
            )
            print(
                "Loaded ImageNet val", len(imagenet_val_loader.dataset), "images", len(imagenet_val_loader), "batches"
            )

        if not self.args.disable_dataloader and self.args.use_videos:
            video_train_loader.set_dataset(
                self.args.dataset(
                    self.args,
                    "train",
                    transform=self.args.transform(self.input_size, "train"),
                    num_images_to_return=self.num_frames,
                ),
                batch_size=self.args.batch_size // self.num_frames,
                shuffle=True,
                collate_fn=self.args.dataset.collate_fn,
                worker_init_fn=self.args.dataset.worker_init_fn,
                drop_last=True,
            )
            print("Loaded Video train", len(video_train_loader.dataset), "images", len(video_train_loader), "batches")

            # Use train transform to make it equally hard.
            video_val_loader.set_dataset(
                self.args.dataset(
                    self.args,
                    "val",
                    transform=self.args.transform(self.input_size, "train"),
                    num_images_to_return=self.num_frames,
                ),
                batch_size=self.args.batch_size // self.num_frames,
                shuffle=True,
                collate_fn=self.args.dataset.collate_fn,
                worker_init_fn=self.args.dataset.worker_init_fn,
            )
            print("Loaded Video val", len(video_val_loader.dataset), "images", len(video_val_loader), "batches")
        print("done in %.3f" % (time.time() - t_start))

    @property
    def iterations_per_epoch(self):
        return self.args.iterations_per_epoch

    def process_imagenet_data(self, data):
        images, labels = data
        data = images[: self.num_frames]
        queue_data = images[self.num_frames:]
        if self.num_frames > 1:
            data = pt_util.remove_dim(torch.stack(data, dim=1), 1)
            queue_data = pt_util.remove_dim(torch.stack(queue_data, dim=1), 1)
            labels = labels.repeat_interleave(self.num_frames)
        else:
            data = data[0]
            queue_data = queue_data[0]
        batch = {
            "data": data,
            "queue_data": queue_data,
            "imagenet_labels": labels,
            "data_source": "IN",
            "num_frames": self.num_frames,
            "batch_type": "images",
            "batch_size": len(data),
        }
        return batch

    def get_imagenet_batch(self, loader_id):
        data = next(self.train_batch_iterators[loader_id])
        self.train_loader_counts[loader_id] += 1
        if self.train_loader_counts[loader_id] == (len(self.train_loaders[loader_id]) + 1):
            # Check this because using never-ending persistent dataloader which never throws stop iteration.
            self.train_loader_counts[loader_id] = 0
            print("Hit ImageNet stop iteration. End of epoch.")
            return None
        return self.process_imagenet_data(data)

    def process_video_data(self, batch):
        data = pt_util.remove_dim(batch["data"], 1)
        queue_data = pt_util.remove_dim(batch["queue_data"], 1)
        batch = {
            "data": data,
            "queue_data": queue_data,
            "data_source": "YT",
            "batch_type": "video",
            "batch_size": len(data),
            "num_frames": self.num_frames,
            "imagenet_labels": torch.full((len(data),), -1, dtype=torch.int64),
        }
        return batch

    def get_video_batch(self, loader_id):
        batch = next(self.train_batch_iterators[loader_id])
        self.train_loader_counts[loader_id] += 1
        if self.train_loader_counts[loader_id] == len(self.train_loaders[loader_id]) + 1:
            # Check this because using never-ending persistent dataloader which never throws stop iteration.
            self.train_loader_counts[loader_id] = 0
            print("Hit Video stop iteration. End of epoch.")
            return None
        return self.process_video_data(batch)

    def setup_other(self):
        t_start = time.time()
        print("Loading CIFAR")
        if self.args.save or self.args.test_first:
            self.cifar_dataset = NPZDataset(
                self.args,
                os.path.join(
                    os.path.dirname(__file__), os.pardir, "datasets", "cifar_data", "cifar_{data_subset}.npz"
                ),
                "train",
                10000,
            )
        else:
            print("Not loading CIFAR, probably in debug")
        print("CIFAR loaded in %.3f" % (time.time() - t_start))

    def setup_optimizer(self):
        base_lr = self.args.base_lr
        params = self.model.parameters()
        param_groups = [{"params": params, "initial_lr": base_lr}]
        optimizer = optim.SGD(param_groups, lr=base_lr, weight_decay=0.0001, momentum=0.9)
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = base_lr
        if self.use_apex:
            (self.model, self.queue_model), optimizer = amp.initialize([self.model, self.queue_model], optimizer,
                                                                       opt_level="O1")

        self.optimizer = optimizer
        self.print_optimizer()

    def setup_model(self):
        torch_devices = self.args.pytorch_gpu_ids
        device = "cuda:" + str(torch_devices[0])
        args = copy.deepcopy(self.args)
        args.title = os.path.join(args.title, "VinceModel")
        args.tensorboard_dir = os.path.join(
            args.base_logdir, args.title, *(args.checkpoint_dir.split(os.sep)[2:-1]), constants.TIME_STR
        )
        args.checkpoint_dir = os.path.join(args.base_logdir, args.title, *(args.checkpoint_dir.split(os.sep)[2:]))
        args.long_save_checkpoint_dir = os.path.join(
            args.base_logdir, args.title, *(args.long_save_checkpoint_dir.split(os.sep)[2:-1]), constants.TIME_STR
        )
        self.model = VinceModel(args)
        print(self.model)
        self.iteration = self.model.restore()
        self.model.to(device)
        self.queue_model = VinceQueueModel(args, self.model)
        self.queue_model.to(device)
        self.vince_queue = StorageQueue(args.vince_queue_size, args.vince_embedding_size, device=device)

        self.epoch = self.iteration // (self.args.iterations_per_epoch * self.args.batch_size)
        if self.iteration > 0:
            print("Resuming epoch", self.epoch)
        self.start_prefetch()
        self.fill_queue_repeat()

    def fill_queue(self):
        # Fill queue with many different batches.
        # Sync parameters because might as well.
        self.queue_model.param_update(self.model, 0)
        num_added = 0
        self.vince_queue.clear()
        print("Filling queue")
        with torch.no_grad():
            pbar = tqdm.tqdm(total=self.vince_queue.maxsize)
            while num_added < self.vince_queue.maxsize:
                batches_concat, batches = self.get_batch()
                outputs = self.queue_model(batches_concat)
                for batch, output in zip(batches, outputs):
                    images = batch["queue_data_cpu"]
                    self.vince_queue.enqueue(output["queue_embeddings"], images, batch["data_source"])
                    num_added += len(images)
                    pbar.update(len(images))
                    if num_added >= self.vince_queue.maxsize:
                        break
            pbar.close()
        print("Queue filled")

    def fill_queue_repeat(self):
        # Fill queue with the same batch over and over.
        # Sync parameters because might as well.
        self.queue_model.param_update(self.model, 0)
        num_added = 0
        self.vince_queue.clear()
        with torch.no_grad():
            batches_concat, batches = self.get_batch()
            outputs = self.queue_model(batches_concat)
            while num_added < self.vince_queue.maxsize:
                for batch, output in zip(batches, outputs):
                    images = batch["queue_data_cpu"]
                    self.vince_queue.enqueue(output["queue_embeddings"], images, batch["data_source"])
                    num_added += len(images)
                    if num_added >= self.vince_queue.maxsize:
                        break
        self.vince_queue.current_tail = 0
        self.vince_queue.full = False
        print("Queue filled with repeats")

    def reset_epoch(self):
        super(VinceSolver, self).reset_epoch()
        self.queue_model.train()
        self.drawn_this_epoch = False

    def prefetch_batches(self):
        while not self.kill_thread:
            batches = []
            for ii in range(len(self.train_batch_fns)):
                loader_id = self.batch_count % len(self.train_batch_fns)
                batch = self.train_batch_fns[loader_id](loader_id)
                if batch is None:
                    self.batch_queue.put(None)
                    batches = None
                    break
                self.batch_count += 1
                initial_images = batch["queue_data"]
                batch = {
                    key: (val.to(self.model.device) if isinstance(val, torch.Tensor) else val)
                    for key, val in batch.items()
                }
                batch["queue_data_cpu"] = initial_images
                batches.append(batch)

            if batches is not None:
                if len(batches) == 1:
                    batches_concat = {
                        key: val if isinstance(val, torch.Tensor) else [val] for key, val in batches[0].items()
                    }
                else:
                    batches_concat = pt_util.stack_dicts_in_list(batches, axis=0, concat=True)
                batches_concat["batch_types"] = batches_concat["batch_type"]
                del batches_concat["batch_type"]
                batches_concat["batch_sizes"] = batches_concat["batch_size"]
                del batches_concat["batch_size"]
                self.batch_queue.put((batches_concat, batches))

    def start_prefetch(self):
        self.prefetch_thread = Thread(target=self.prefetch_batches)
        self.prefetch_thread.start()

    def end(self):
        self.kill_thread = True

    def get_batch(self):
        batches = self.batch_queue.get()
        while batches is None:
            self.fill_queue_repeat()
            batches = self.batch_queue.get()
        return batches

    def run_train_iteration(self):
        total_t_start = time.time()
        t_start = time.time()
        image_batch_concat, image_batches = self.get_batch()

        t_end = time.time()
        self.time_meters["data_cache_time"].update(t_end - t_start)
        t_start = time.time()

        # Feed in whole batch together through encoder (the most computationally expensive part).

        if self.args.jigsaw:
            if random.random() < 0.5:
                queue_batches = self.queue_model(image_batch_concat, jigsaw=True, shuffle=True)
                outputs = self.model.get_embeddings(image_batch_concat, jigsaw=False, shuffle=True)
            else:
                queue_batches = self.queue_model(image_batch_concat, jigsaw=False, shuffle=True)
                outputs = self.model.get_embeddings(image_batch_concat, jigsaw=True, shuffle=True)
        else:
            queue_batches = self.queue_model(image_batch_concat, shuffle=True)
            outputs = self.model.get_embeddings(image_batch_concat, shuffle=True)

        t_end = time.time()
        self.time_meters["forward_time"].update(t_end - t_start)
        t_start = time.time()

        loss_list = []
        metrics_list = []
        # Feed in batch as separate mini-batches depending on batch type.
        image_batches = self.model.split_dict_by_type(
            image_batch_concat["batch_types"], image_batch_concat["batch_sizes"], image_batch_concat
        )

        for bb, (image_batch, queue_batch, output) in enumerate(zip(image_batches, queue_batches, outputs)):
            output.update(self.vince_queue.dequeue())
            output.update(image_batch)
            output.update(queue_batch)

            output.update(self.model(output))
            loss_dict = self.model.loss(output)
            metrics = self.model.get_metrics(output)
            loss_list.append({key: val[0] * val[1] for key, val in loss_dict.items()})
            metrics_list.append(metrics)

        loss_dict = pt_util.stack_dicts_in_list(loss_list)
        loss_dict = {key: val.mean() for key, val in loss_dict.items()}
        metrics = pt_util.stack_dicts_in_list(metrics_list)
        metrics = {key: val.mean() for key, val in metrics.items()}

        updated_loss_meters = set()
        try:
            total_loss = 0
            for key, weighted_loss in loss_dict.items():
                total_loss = total_loss + weighted_loss
                self.loss_meters[key].update(weighted_loss)
                updated_loss_meters.add(key)
            if "total_loss" in self.loss_meters:
                self.loss_meters["total_loss"].update(total_loss)
                updated_loss_meters.add("total_loss")
            loss = total_loss
            assert torch.isfinite(loss)
        except AssertionError as re:
            import pdb
            traceback.print_exc()
            pdb.set_trace()
            print("anomoly", re)
            raise re

        updated_metric_meters = set()
        for key, val in metrics.items():
            self.metric_meters[key].update(val)
            updated_metric_meters.add(key)

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

        for bb, (image_batch, output) in enumerate(zip(image_batches, outputs)):
            if not self.drawn_this_epoch and self.vince_queue.full:
                # If we haven't drawn yet and the queue is full of many different batches, draw now.
                if bb == len(image_batches) - 1:
                    self.drawn_this_epoch = True
                print("Drawing Tensorboard images")
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

            # Must be done after image logger because images are reassigned in place.
            # update queue and queue params
            queue_images_cpu = image_batch["queue_data_cpu"]
            self.vince_queue.enqueue(output["queue_embeddings"], queue_images_cpu, image_batch["data_source"])

        self.queue_model.vince_update(self.model)
        if self.logger_iteration % self.args.save_frequency == 0:
            self.model.save(self.iteration, 5)

        if self.logger_iteration % self.args.log_frequency == 0:
            log_dict = {"times/%s/%s" % (self.full_name, key): val.val for key, val in self.time_meters.items()}
            log_dict.update(
                {"losses/%s/%s" % (self.full_name, key): self.loss_meters[key].val for key in updated_loss_meters}
            )
            log_dict.update(
                {"metrics/%s/%s" % (self.full_name, key): self.metric_meters[key].val for key in updated_metric_meters}
            )
            if self.train_logger is not None:
                self.train_logger.dict_log(log_dict, self.iteration)

        self.iteration += self.args.batch_size

        total_t_end = time.time()
        self.time_meters["total_time"].update(total_t_end - total_t_start)
        self.logger_iteration += 1

    def run_val(self):
        with torch.no_grad():
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

            updated_epoch_loss_meters = set()
            updated_epoch_metric_meters = set()

            step_on = self.iteration

            for val_name, val_loader, data_processor in zip(self.val_data_names, self.val_loaders, self.val_batch_fns):
                print("Running val for", val_name)
                total_t_start = time.time()
                test_t_start = time.time()
                for ii, image_batch in enumerate(tqdm.tqdm(val_loader)):
                    if test_t_start - time.time() > 5 * 60:
                        # Break after 5 minutes.
                        break
                    image_batch = data_processor(image_batch)
                    image_batch["batch_types"] = [image_batch["batch_type"]]
                    del image_batch["batch_type"]
                    image_batch["batch_sizes"] = [image_batch["batch_size"]]
                    del image_batch["batch_size"]
                    image_batch = {
                        key: (val.to(self.model.device, non_blocking=True) if isinstance(val, torch.Tensor) else val)
                        for key, val in image_batch.items()
                    }

                    batch_size = image_batch["data"].shape[0]

                    t_end = time.time()
                    time_meters["data_cache_time"].update(t_end - total_t_start)
                    t_start = time.time()

                    image_batch.update(self.queue_model(image_batch)[0])
                    image_batch.update(self.model.get_embeddings(image_batch)[0])
                    image_batch.update(self.vince_queue.dequeue())

                    image_batch.update(self.model(image_batch))
                    output = image_batch
                    loss_dict = self.model.loss(output)

                    t_end = time.time()
                    time_meters["forward_time"].update(t_end - t_start)
                    t_start = time.time()

                    metrics = self.model.get_metrics(output)
                    if ii % self.args.image_log_frequency == 0:
                        image_output = self.model.get_image_output(output)

                    updated_loss_meters = set()
                    total_loss = 0
                    for key, val in loss_dict.items():
                        weighted_loss = val[0] * val[1]
                        total_loss = total_loss + weighted_loss
                        loss_meters[key].update(weighted_loss)
                        epoch_loss_meters["epoch_" + key].update(weighted_loss, batch_size)
                        updated_loss_meters.add(key)
                        updated_epoch_loss_meters.add("epoch_" + key)
                    if "total_loss" in loss_meters:
                        loss_meters["total_loss"].update(total_loss)
                        epoch_loss_meters["epoch_total_loss"].update(total_loss, batch_size)
                        updated_loss_meters.add("total_loss")
                        updated_epoch_loss_meters.add("epoch_total_loss")
                    loss = total_loss

                    try:
                        assert torch.isfinite(loss)
                    except:
                        # output = self.model.forward(image_batch)
                        print("Nan loss", loss_dict)

                    updated_metric_meters = set()
                    for key, val in metrics.items():
                        metric_meters[key].update(val)
                        updated_metric_meters.add(key)
                        epoch_metric_meters["epoch_" + key].update(val, batch_size)
                        updated_epoch_metric_meters.add("epoch_" + key)

                    t_end = time.time()
                    time_meters["metrics_time"].update(t_end - t_start)

                    if ii % self.args.image_log_frequency == 0:
                        if self.val_logger is not None:
                            for key, val in image_output.items():
                                if isinstance(val, list):
                                    for vv, item in enumerate(val):
                                        self.val_logger.image_summary(
                                            self.full_name + "_" + key[len("images/"):], item, step_on + vv, False
                                        )
                                else:
                                    self.val_logger.image_summary(
                                        self.full_name + "_" + key[len("images/"):], val, step_on, False
                                    )

                    if ii % self.args.log_frequency == 0:
                        log_dict = {"times/%s/%s" % (self.full_name, key): val.val for key, val in time_meters.items()}
                        log_dict.update(
                            {
                                "losses/%s/%s" % (self.full_name, key): loss_meters[key].val
                                for key in updated_loss_meters
                            }
                        )
                        log_dict.update(
                            {
                                "metrics/%s/%s" % (self.full_name, key): metric_meters[key].val
                                for key in updated_metric_meters
                            }
                        )
                        if self.val_logger is not None:
                            self.val_logger.dict_log(log_dict, step_on)

                    step_on += self.args.batch_size
                    total_t_end = time.time()
                    time_meters["total_time"].update(total_t_end - total_t_start)
                    total_t_start = time.time()

            ##### CIFAR #####
            epoch_metric_meters["epoch_knn_cifar"] = AverageMeter()

            all_features = []
            imagenet_mean = pt_util.from_numpy(constants.IMAGENET_MEAN).to(self.model.device).view(1, -1, 1, 1)
            imagenet_std = pt_util.from_numpy(constants.IMAGENET_STD).to(self.model.device).view(1, -1, 1, 1)

            print("Running CIFAR")
            for start_ind in tqdm.tqdm(range(0, len(self.cifar_dataset), self.args.batch_size)):
                data = self.cifar_dataset.data[
                       start_ind: min(len(self.cifar_dataset), start_ind + self.args.batch_size)
                       ]
                data = data.to(device=self.model.device, dtype=torch.float32)
                data = data - imagenet_mean
                data.div_(imagenet_std)
                features = self.model.get_embeddings({"data": data})["embeddings"]
                all_features.append(pt_util.to_numpy(features))
            all_images = np.transpose(pt_util.to_numpy(self.cifar_dataset.data), (0, 2, 3, 1))
            labels = pt_util.to_numpy(self.cifar_dataset.labels)
            all_features = np.concatenate(all_features, axis=0)
            if len(all_features.shape) == 4:
                # all_features = pt_util.remove_dim(all_features, dim=(2, 3))
                all_features = np.mean(all_features, axis=(2, 3))

            if self.val_logger is not None:
                kdt = KDTree(all_features, leaf_size=40, metric="euclidean")
                neighbors = kdt.query(all_features, k=11)[1]
                # remove self match
                neighbors = neighbors[:, 1:]
                preds_all = labels[neighbors]
                preds = scipy.stats.mode(preds_all, axis=1)[0].squeeze(1)
                acc = np.mean(preds == labels)
                epoch_metric_meters["epoch_knn_cifar"].update(acc)
                updated_epoch_metric_meters.add("epoch_knn_cifar")

                nn_inds = kdt.query(all_features[0:100:10], k=10)[1]
                image = drawing.subplot(
                    all_images[nn_inds.reshape(-1)], 10, 10, self.args.input_width, self.args.input_height, border=10
                )

                self.val_logger.image_summary(
                    self.full_name + "_kNN/cifar", image, step_on, increment_counter=False, max_size=1000
                )

        log_dict = {
            "epoch/losses/%s/%s" % (self.full_name, key): epoch_loss_meters[key].avg
            for key in updated_epoch_loss_meters
        }
        log_dict.update(
            {
                "epoch/metrics/%s/%s" % (self.full_name, key): epoch_metric_meters[key].avg
                for key in updated_epoch_metric_meters
            }
        )
        if self.val_logger is not None:
            self.val_logger.dict_log(log_dict, step_on)
