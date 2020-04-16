import time

import cv2
import numpy as np

from datasets.base_dataset import BaseDataset


class PairDataset(BaseDataset):
    def __init__(self, args, seqs, data_subset="train", pair_transform=None, transforms=None, pairs_per_seq=25):
        super(PairDataset, self).__init__(args, data_subset)
        self.seqs = seqs
        self.pair_transform = pair_transform
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))
        self.length = np.sum(self.indices)
        self.return_meta = getattr(seqs, "return_meta", False)
        self.seq_sizes = {}
        self.invalid_seqs = {}

    def __getitem__(self, index):
        t_start = time.time()
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get("cover", None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        # filter out noisy frames

        val_indices = self._filter(img_files[0], index, anno, vis_ratios)
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        rand_z, rand_x = self._sample_pair(val_indices)

        box_z = anno[rand_z]
        box_x = anno[rand_x]

        z = cv2.imread(img_files[rand_z])[:, :, ::-1]
        x = cv2.imread(img_files[rand_x])[:, :, ::-1]
        # z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        # print('box x', box_x, 'box z', box_z)

        item = (z, x, box_z, box_x)
        t_end = time.time()
        # if self.worker_id == 0:
        # print('reading image time', t_end - t_start)
        if self.pair_transform is not None:
            t_start = time.time()
            exemplar_img, track_img = self.pair_transform(item)
            t_end = time.time()
            # if self.worker_id == 0:
            # print('first transform image time', t_end - t_start)
            t_start = time.time()
            # if t_end - t_start > 0.5:
            # print('pair transform', (t_end - t_start))
            # print('item', exemplar_img.shape, track_img.shape)
            if self.transforms is not None:
                exemplar_img = self.transforms(exemplar_img)
                track_img = self.transforms(track_img)
            item = (exemplar_img, track_img)
            t_end = time.time()
            # if self.worker_id == 0:
            # print('second transform image time', t_end - t_start)

        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    # if 30 < abs(rand_x - rand_z) < 500:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x

    def _filter(self, img0, key, anno, vis_ratios=None):
        if key in self.invalid_seqs:
            return self.invalid_seqs[key]
        if key not in self.seq_sizes:
            self.seq_sizes[key] = cv2.imread(img0).shape[:2]
        size = self.seq_sizes[key]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = vis_ratios > max(1, vis_ratios.max() * 0.3)
        else:
            c8 = np.ones_like(c1)

        mask = np.logical_and.reduce((c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]
        if len(val_indices) < 2:
            self.invalid_seqs[key] = val_indices
        return val_indices
