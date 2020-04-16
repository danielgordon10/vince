import copy
import glob
import os
import random
import time
from typing import Tuple

import cv2
import numpy as np
import tqdm
from dg_util.python_utils import misc_util
from dg_util.python_utils import video_utils
from dg_util.python_utils import youtube_utils

import constants
from datasets.base_dataset import BaseDataset

SAMPLE_NAME = "AA/AA2pFq9pFTA_000001.jpg"
LEN_SAMPLE_NAME = len(SAMPLE_NAME)
LEN_VID_NAME = len("AA2pFq9pFTA")
LEN_NUM_NAME = len("000001")


class VideoCacherDataset(BaseDataset):
    @staticmethod
    def get_video_name(name):
        return name[-LEN_NUM_NAME - LEN_VID_NAME - 5: -LEN_NUM_NAME - 5]

    @staticmethod
    def get_frame_id(name):
        return int(name[-LEN_NUM_NAME - 4: -4])

    def video_id_frame_id_split(self, name):
        return self.get_video_name(name), self.get_frame_id(name)

    def get_image_paths(self):
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.output_path, "*/*.jpg")))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.output_path, key[:2], key + "_%06d.jpg" % ind)

    def __init__(
            self,
            args,
            video_ids,
            data_subset: str = "train",
            ignore_set=None,
            filter_similar_frames=True,
            filter_using_laplacian=True,
            filter_using_flow=False,
    ):
        super(VideoCacherDataset, self).__init__(args, data_subset)
        if ignore_set is None:
            ignore_set = set()
        elif not isinstance(ignore_set, set):
            ignore_set = set(ignore_set)
        self.uncached_ids = []
        self.num_frames_to_cache = self.args.num_frames
        self.max_video_length = self.args.max_video_length
        self.output_path = os.path.join(self.args.data_path, data_subset)
        self.filter_similar_frames = filter_similar_frames
        self.filter_using_laplacian = filter_using_laplacian
        self.filter_using_flow = filter_using_flow
        prev_images = self.get_image_paths()
        prev_vid_ids = [self.get_video_name(img) for img in prev_images]
        prev_vid_ids = set(prev_vid_ids)
        print("previously cached", len(prev_vid_ids))
        if len(ignore_set) > 0:
            print("ignore size", len(ignore_set))
        print("total dataset size", len(video_ids))
        self.uncached_ids = set(video_ids) - prev_vid_ids
        self.uncached_ids -= ignore_set
        print("uncached dataset size", len(self.uncached_ids))
        self.uncached_ids = sorted(self.uncached_ids)

    def __len__(self):
        return len(self.uncached_ids)

    # C++ style return codes where 0 is success, >0 is issue.
    def __getitem__(self, idx) -> Tuple[str, int]:
        time.sleep(0.01)
        video_id = self.uncached_ids[idx]
        video = youtube_utils.download_video(
            video_id, video_path="/tmp/downloaded_videos", cookie_path=constants.COOKIE_PATH
        )
        if video is None:
            print("no video", youtube_utils.get_video_url(video_id))
            return video_id, 1
        found_frames = False
        mean_num_frames = []
        errors = set()
        # Try a few times because downloading the video is the slow part.
        for _ in range(10):
            frames, frame_inds = video_utils.get_frames(
                video,
                self.args.video_sample_rate,
                remove_video=False,
                max_frames=self.args.max_video_length,
                return_inds=True,
            )
            if frames is None or len(frames) == 0:
                errors.add("no frames")
                continue

            if self.filter_using_flow:
                prev_frames = copy.deepcopy(frames[:-1])
                frames = frames[1:]
                frame_inds = frame_inds[1:]
            if self.filter_similar_frames:
                frames, inds = video_utils.filter_similar_frames(frames, return_inds=True)
                if len(frames) < self.num_frames_to_cache:
                    mean_num_frames.append(len(frames))
                    errors.add("similar")
                    continue
                frame_inds = frame_inds[inds]
                if self.filter_using_flow:
                    prev_frames = list([prev_frames[ind] for ind in inds])

            if len(frames) > self.max_video_length > 0:
                rand_start_point = random.randint(0, len(frames) - self.max_video_length - 1)
                frames = frames[rand_start_point: rand_start_point + self.max_video_length]
                if self.filter_using_flow:
                    prev_frames = prev_frames[rand_start_point: rand_start_point + self.max_video_length]
                frame_inds = frame_inds[rand_start_point: rand_start_point + self.max_video_length]

            frames, inds = video_utils.remove_border(frames, return_inds=True)
            if frames[0].shape[0] == 0 or frames[0].shape[1] == 0:
                errors.add("border")
                continue
            frames = np.stack(frames, axis=0)
            if self.filter_using_flow:
                prev_frames = [prev_frame[inds[2]: inds[3], inds[0]: inds[1]] for prev_frame in prev_frames]
                prev_frames = np.stack(prev_frames, axis=0)

            if self.filter_using_laplacian:
                frames, inds = video_utils.filter_using_laplacian(frames, return_inds=True)
                if len(frames) < self.num_frames_to_cache:
                    mean_num_frames.append(len(frames))
                    errors.add("laplacian")
                    continue
                frame_inds = frame_inds[inds]
                if self.filter_using_flow:
                    prev_frames = prev_frames[inds]

            if self.filter_using_flow:
                frames, flow_masks, inds = video_utils.filter_using_flow(prev_frames, frames, return_inds=True)
                if len(frames) < self.num_frames_to_cache:
                    mean_num_frames.append(len(frames))
                    errors.add("flow")
                    continue
                frame_inds = frame_inds[inds]

            found_frames = True
            break

        os.remove(video)

        if not found_frames:
            if len(mean_num_frames) == 0:
                mean_num_frames = [0]
            print(
                "filter returned %.1f frames %s %s"
                % (np.mean(mean_num_frames), ", ".join(sorted(list(errors))), youtube_utils.get_video_url(video_id))
            )
            return video_id, 2

        if self.args.only_use_shots:
            shots, shot_borders = video_utils.get_shots(frames, return_inds=True)
            shot_frame_ids = [frame_inds[shot_borders[ii]: shot_borders[ii + 1]] for ii in range(len(shot_borders))]
        else:
            shots = [frames]
            shot_frame_ids = [frame_inds]

        shots_long_enough = [ii for ii, shot in enumerate(shots) if len(shot) >= self.num_frames_to_cache]
        if len(shots_long_enough) == 0:
            print("no shots long enough", youtube_utils.get_video_url(video_id))
            return video_id, 3

        shots = [shots[ii] for ii in shots_long_enough]
        shot_frame_ids = [shot_frame_ids[ii] for ii in shots_long_enough]
        max_num_shots = len(shots) if self.args.max_num_shots <= 0 else self.args.max_num_shots
        for cc in range(max_num_shots):
            frames = shots[cc]
            frame_inds = shot_frame_ids[cc]
            start_point = random.randint(0, len(frames) - self.num_frames_to_cache)
            frames = frames[start_point: start_point + self.num_frames_to_cache]
            self.cache_images(frames, video_id, frame_inds, num_frames=-1, max_side_size=self.args.max_side_size)
        return video_id, 0

    def cache_images(self, frames, vid_id, frame_inds, num_frames=-1, max_side_size=-1, min_side_size=-1):
        assert not ((max_side_size > 0) and (min_side_size > 0)), "Can only specify max_side_size or min_side_size."
        im_dir = os.path.join(self.output_path, vid_id[:2])
        os.makedirs(im_dir, exist_ok=True)
        if num_frames < 1:
            random_inds = np.arange(len(frames))
        else:
            random_inds = np.random.choice(len(frames), min(num_frames, len(frames)), replace=False)
            random_inds.sort()

        for rand_ind in random_inds:
            image = frames[rand_ind]
            image_id = frame_inds[rand_ind]
            if len(image.shape) != 3:
                print("image is wrong shape", vid_id, image.shape)
                return
            if max_side_size > 0:
                image = misc_util.max_resize(image, max_side_size, always_resize=False)
            elif min_side_size > 0:
                image = misc_util.min_resize(image, max_side_size, always_resize=False)
            cv2.imwrite(os.path.join(im_dir, "%s_%06d.jpg" % (vid_id, image_id)), image[:, :, ::-1])
