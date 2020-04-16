import numpy as np
import torch
from dg_util.python_utils import misc_util
from dg_util.python_utils import pytorch_util as pt_util

from dg_util.python_utils.tensor_dataset import TensorDataset


class NPZDataset(TensorDataset):
    """
    Convenience class for fast reading of saved numpy image arrays without the need for slicing and concating.
    """

    def __init__(self, args, path, data_subset, num_data_points=None, contiguous=True):
        with torch.no_grad():
            self.args = args
            self.data_subset = data_subset
            npz_dataset = np.load(path.format(data_subset=data_subset))
            data = npz_dataset["data"]
            labels = pt_util.from_numpy(npz_dataset["labels"])
            if num_data_points is None:
                num_data_points = len(data)

            if num_data_points < len(data):
                np.random.seed(0)
                rand_inds = np.random.choice(len(data), num_data_points, replace=False)
                data = data[rand_inds]
                labels = labels[rand_inds]

            assert len(data.shape) == 4

            if data.shape[1] == 3:
                data = data.transpose(0, 2, 3, 1)

            data = misc_util.resize(data, (args.input_width, args.input_height), height_channel=1, width_channel=2)

            data = pt_util.from_numpy(data).permute(0, 3, 1, 2)
            if contiguous:
                data = data.contiguous()
            super(NPZDataset, self).__init__(data, labels, self.args.batch_size)
