import glob
import os

import tqdm
from torch.utils.data import DataLoader

import arg_parser
from datasets.video_cacher_dataset import VideoCacherDataset

"""
Example run command

python run_cache_video_dataset.py --title cache --description caching --num-workers 100
"""


def main():
    args = arg_parser.parse_args()

    data_subset = "val"
    args.data_path = "/home/xkcd/datasets/r2v2_large_with_ids_debug"

    # Good Params
    args.num_frames_to_cache = 4
    args.video_sample_rate = 150
    args.max_video_length = 10
    args.only_use_shots = False
    args.max_side_length = 480

    # Other stuff
    args.batch_size = 1

    # Stops MKL from taking lots of memory because of resizes and such (torch kernels).
    os.environ["LRU_CACHE_CAPACITY"] = "1"

    urls = sorted(glob.glob("youtube_scrape/urls*%s.csv" % data_subset), key=os.path.getmtime)
    video_ids = [line.strip().split('"')[1] for line in open(urls[-1])]

    ignore_set = []
    if os.path.exists(os.path.join(args.data_path, "failed_video_ids.txt")):
        ignore_set.extend([line.strip() for line in open(os.path.join(args.data_path, "failed_video_ids.txt"))])
    if os.path.exists(os.path.join(args.data_path, "no_images_video_ids.txt")):
        ignore_set.extend([line.strip() for line in open(os.path.join(args.data_path, "no_images_video_ids.txt"))])

    yt_dataset = VideoCacherDataset(args, video_ids, data_subset, ignore_set=ignore_set)

    train_loader = DataLoader(
        yt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False
    )

    os.makedirs(args.data_path, exist_ok=True)
    failed_video_ids_file = open(os.path.join(args.data_path, "failed_video_ids.txt"), "a+")
    no_image_video_ids_file = open(os.path.join(args.data_path, "no_images_video_ids.txt"), "a+")

    for data in tqdm.tqdm(train_loader):

        for video_id, error_code in zip(*data):
            if error_code == 0:
                continue
            elif error_code == 1:
                failed_video_ids_file.write(video_id + "\n")
                failed_video_ids_file.flush()
            elif error_code > 1:
                no_image_video_ids_file.write(video_id + "\n")
                no_image_video_ids_file.flush()


if __name__ == "__main__":
    main()
