import abc
import os
import pickle

import numpy as np

import constants
from datasets.base_dataset import BaseDataset


class BaseMultiFrameDataset(BaseDataset):
    def __init__(self, args, path_info, data_subset: str = "train", transform=None, num_images_to_return=-1):
        super(BaseMultiFrameDataset, self).__init__(args, data_subset)
        if num_images_to_return <= 0:
            self.num_images_to_return = self.args.num_frames
        else:
            self.num_images_to_return = num_images_to_return

        print("Loaded %s dataset with %d entries." % (data_subset, len(path_info)))
        print("Filtering for videos with length", self.num_images_to_return)
        self.path_info = path_info
        self.path_info = list(filter(lambda x: len(x[1]) >= self.num_images_to_return, self.path_info))
        num_frames = int(np.sum([len(p_info[1]) for p_info in self.path_info]))
        print("Num for %s videos %d frames %d" % (data_subset, len(self.path_info), num_frames))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.path_info)


class VideoDataset(BaseMultiFrameDataset, abc.ABC):
    def __init__(self, args, data_subset, transform, num_images_to_return=-1, check_for_new_data=False):
        self.data_basepath = args.data_path
        self.data_split_path = os.path.join(self.data_basepath, data_subset)
        pickle_path = os.path.join(self.data_basepath, data_subset + "_names.pkl")
        if not os.path.exists(pickle_path) or constants.CHECK_FOR_NEW_DATA or check_for_new_data:
            images = self.get_image_paths()
            path_info = {}
            video_names = sorted([self.video_id_frame_id_split(name) for name in images])
            for vid_id, ind in video_names:
                if vid_id not in path_info:
                    path_info[vid_id] = []
                path_info[vid_id].append(ind)
            path_info = sorted([(key, val) for key, val in path_info.items()])
            os.makedirs(self.data_split_path, exist_ok=True)
            pickle.dump(path_info, open(pickle_path, "wb"))
        path_info = pickle.load(open(pickle_path, "rb"))
        super(VideoDataset, self).__init__(
            args, path_info, data_subset, transform=transform, num_images_to_return=num_images_to_return
        )

    @staticmethod
    def get_video_name(name):
        raise NotImplementedError

    @staticmethod
    def get_frame_id(name):
        raise NotImplementedError

    def video_id_frame_id_split(self, name):
        return self.get_video_name(name), self.get_frame_id(name)

    def get_image_paths(self):
        raise NotImplementedError

    def get_image_name(self, key: str, ind: int):
        raise NotImplementedError
