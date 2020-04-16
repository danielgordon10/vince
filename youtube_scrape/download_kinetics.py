"""
python run_download_kinetics.py
"""

import csv
import glob
import multiprocessing as mp
import os
import random
import traceback

import cv2
import tqdm
from dg_util.python_utils import misc_util
from dg_util.python_utils import video_utils
from dg_util.python_utils import youtube_utils

NUM_THREADS = 50
FPS = 10
MAX_SIDE_SIZE = 320
dataset_folder = "/home/xkcd/datasets/kinetics400_data/"

annotation_folder = os.path.join(dataset_folder, "annotations")
image_folder = os.path.join(dataset_folder, "images", "%dfps" % FPS)


def _read_csv(csv_file):
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        all_rows = [dict(row) for row in reader]
        keys = reader.fieldnames
    return keys, all_rows


def download_youtube_video(args):
    row, subset_name = args
    video_id = row["youtube_id"]
    time_start = int(row["time_start"])
    time_end = int(row["time_end"])
    video_path = youtube_utils.download_video(video_id, "/tmp/ytdl/")
    if not video_path:
        print("Failed_to_download", video_id)
        return
    # Storing each video in its own folder is bad for the OS. Storing them all in one folder is also bad.
    # Storing them in 4096 subfolders seems to be good.
    path_for_images = os.path.join(image_folder, subset_name, video_id[:2])
    os.makedirs(path_for_images, exist_ok=True)
    try:
        frames = video_utils.get_frames_by_time(video_path, start_time=time_start, end_time=time_end, fps=FPS)
        remainder = (time_end - time_start) * FPS - len(frames)
        if remainder > 0:
            # Sometimes the annotations are wrong and give timestamps beyond the end of the video.
            time_start = max(0, time_start - int(float(remainder) / FPS))
            time_end = time_end + int(remainder / FPS)
            frames = video_utils.get_frames_by_time(video_path, start_time=time_start, end_time=time_end, fps=FPS)
        # remove the black boundaries from YouTube videos that may be automatically added
        frames = video_utils.remove_border(frames)

        # resize the images and save them
        for frame_ind, frame in enumerate(frames):
            frame = misc_util.max_resize(frame, MAX_SIDE_SIZE, always_resize=False)
            cv2.imwrite(os.path.join(path_for_images, "%s_%06d.jpg" % (video_id, frame_ind)), frame[:, :, ::-1])
    except Exception:
        traceback.print_exc()
    finally:
        os.remove(video_path)


def download_youtube_videos(data_dict_rows, subset_name):
    pool = mp.Pool(NUM_THREADS)

    images = glob.glob(os.path.join(image_folder, subset_name, "*/*.jpg"))
    images += glob.glob(os.path.join(dataset_folder, "images", "%sfps" % FPS, subset_name, "*/*.jpg"))

    done_ids = set()
    for image in images:
        vid_id = image[-22:-11]
        done_ids.add(vid_id)

    print("starting len", len(data_dict_rows))
    print("done", len(done_ids))
    del images
    data_dict_rows = list(filter(lambda row: row["youtube_id"] not in done_ids, data_dict_rows))
    del done_ids
    print("new len", len(data_dict_rows))
    video_args = list(zip(data_dict_rows, [subset_name] * len(data_dict_rows)))
    random.shuffle(video_args)
    list(tqdm.tqdm(pool.imap(download_youtube_video, video_args), total=len(video_args)))


def main():
    # for subset_name in ['train', 'validate', 'test']:
    for subset_name in ["validate", "test"]:
        print("subset", subset_name)
        keys, all_rows = _read_csv(os.path.join(annotation_folder, subset_name + ".csv"))
        os.makedirs(os.path.join(image_folder, subset_name), exist_ok=True)
        download_youtube_videos(all_rows, subset_name)


if __name__ == "__main__":
    main()
