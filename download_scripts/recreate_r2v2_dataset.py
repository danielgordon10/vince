import multiprocessing as mp
import os
import threading
from typing import List, Tuple

import cv2
import tqdm
from dg_util.python_utils import video_utils, youtube_utils

COOKIE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, "youtube_scrape", "cookies.txt"))
LEN_NUM_NAME = len("000001")

SAVE_DIRECTORY = os.path.abspath("r2v2_dataset")


def parse_ids_file(dataset_path: str) -> Tuple[List[str], List[List[int]]]:
    with open(dataset_path) as vid_fi:
        urls = []
        frames = []
        for line in vid_fi.readlines():
            parts = line.split(",")
            urls.append(parts[0])
            frames.append(list(map(int, parts[1:])))

    return urls, frames


def download_video(video_id: str):
    try:
        video_path = youtube_utils.download_video(
            video_id, video_path="/tmp/downloaded_videos", cookie_path=COOKIE_PATH
        )
    except:
        return None
    return video_path


def save_images_thread(queue: mp.Queue, dataset_dir: str):
    video_data = -1
    while video_data is not None:
        video_data = queue.get()
        video_id, frame_inds, video_path = video_data
        frames = []
        for _ in range(10):
            # Try 10 times because sometimes frame extraction can fail
            vidcap = cv2.VideoCapture(video_path)
            frames = []
            for frame_ind in frame_inds:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
                success, image = vidcap.read(1)
                if not success:
                    break
                frames.append(image)
            if len(frames) == len(frame_inds):
                break

        if len(frames) > 0:
            frames = video_utils.remove_border(frames)
            os.makedirs(os.path.join(dataset_dir, video_id[:2]), exist_ok=True)
            for frame, frame_ind in zip(frames, frame_inds):
                cv2.imwrite(
                    os.path.join(dataset_dir, video_id[:2], f"{video_id}_{frame_ind:0{LEN_NUM_NAME}d}.jpg"), frame
                )
            if len(frames) > 0:
                # Remove the raw video since we're done with it now
                os.remove(video_path)


def try_download_and_save_frames(args):
    video_id, frame_inds, dataset_dir, queue = args
    if os.path.exists(os.path.join(dataset_dir, video_id[:2], f"{video_id}_{frame_inds[-1]:0{LEN_NUM_NAME}d}.jpg")):
        # Skip, already got this data
        print(f"Skipping {video_id}, already have data downloaded")
        return True
    video_path = download_video(video_id)
    if video_path is not None:
        queue.put((video_id, frame_inds, video_path))
        return True
    return False


def download_dataset(urls: List[str], frames: List[List[int]], dataset_dir: str):
    with mp.Manager() as manager:
        pool = mp.Pool(mp.cpu_count())
        queue = manager.Queue(1)
        parallel_args = list(zip(urls, frames, [dataset_dir] * len(urls), [queue] * len(urls)))
        thread = threading.Thread(target=save_images_thread, daemon=True, args=(queue, dataset_dir))
        thread.start()
        successful_download = list(
            tqdm.tqdm(pool.imap(try_download_and_save_frames, parallel_args), total=len(parallel_args))
        )
        thread.join()
        return successful_download


def main():
    dataset_dir = os.path.join(SAVE_DIRECTORY, "r2v2_train")
    os.makedirs(dataset_dir, exist_ok=True)
    urls, frames = parse_ids_file("datasets/info_files/r2v2_ids_train.txt")
    download_dataset(urls, frames, dataset_dir)

    dataset_dir = os.path.join(SAVE_DIRECTORY, "r2v2_val")
    os.makedirs(dataset_dir, exist_ok=True)
    urls, frames = parse_ids_file("datasets/info_files/r2v2_ids_val.txt")
    download_dataset(urls, frames, dataset_dir)


if __name__ == "__main__":
    main()
