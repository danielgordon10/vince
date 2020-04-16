import cv2
import numpy as np
import torch
import tqdm
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from dg_util.python_utils import tsne
from torch.utils.data import DataLoader

import arg_parser
import constants
from datasets import R2V2Dataset
from models.vince_model import VinceModel
from utils.transforms import StandardVideoTransform

"""
Example run command
python visualizations/dataset_mosaic.py \
    --title sample_mosaic \
    --description none \
    --checkpoint-dir logs/moco/MocoImagenetModel/checkpoints_r18-b-256-q-65536-fsize-64-vid-ibc-4-no-self/ \
    --data-path /home/xkcd/datasets/r2v2_large_with_ids/ \
    --num-workers 20 --backbone ResNet18 --pytorch-gpu-ids 0 --feature-extractor-gpu-ids 0 \
    --num-frames 4 -b 8 \
"""

NUM_IMAGES_PER_ROW = 16
NUM_IMAGES_IN_TSNE = 16384


def make_mean_and_std(inputs, mean, std):
    _, mean = pt_util.fix_broadcast(inputs, mean)
    _, std = pt_util.fix_broadcast(inputs, std)
    mean = mean.to(dtype=inputs.dtype, device=inputs.device)
    std = std.to(dtype=inputs.dtype, device=inputs.device)
    return mean, std


def to_uint8(images):
    images = pt_util.from_numpy(images).squeeze(0)
    images.mul_(std)
    images.add_(mean)
    images = pt_util.to_numpy(images.to(torch.uint8)).transpose(0, 2, 3, 1)
    return images


def process_video_data(batch):
    data = pt_util.remove_dim(batch["data"], 1)
    queue_data = pt_util.remove_dim(batch["queue_data"], 1)
    batch = {
        "data": data,
        "queue_data": queue_data,
        "data_source": "YT",
        "batch_type": "video",
        "batch_size": len(data),
        "num_frames": args.num_frames,
        "imagenet_labels": torch.full((len(data),), -1, dtype=torch.int64),
    }
    return batch


args = arg_parser.parse_args()

assert NUM_IMAGES_PER_ROW % args.num_frames == 0

torch_devices = args.pytorch_gpu_ids

print("starting mosaic")
inputs = torch.zeros(2, 3, args.input_height, args.input_width, dtype=torch.float32)
mean = pt_util.from_numpy(constants.IMAGENET_MEAN)
std = pt_util.from_numpy(constants.IMAGENET_STD)
mean, std = make_mean_and_std(inputs, mean, std)

dataset = R2V2Dataset(
    args, "val", transform=StandardVideoTransform(args.input_size, "val"), num_images_to_return=args.num_frames
)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=False,
    collate_fn=R2V2Dataset.collate_fn,
)

all_images = []

for data in tqdm.tqdm(data_loader, total=NUM_IMAGES_PER_ROW ** 2 // args.num_frames):
    images = to_uint8(data["data"])
    for image in images:
        all_images.append(image)
    if len(all_images) >= NUM_IMAGES_PER_ROW ** 2:
        break

del data_loader

mosaic = drawing.subplot(
    all_images, NUM_IMAGES_PER_ROW, NUM_IMAGES_PER_ROW, args.input_width, args.input_height, border=5
)
cv2.imwrite("mosaic_%s.jpg" % args.title, mosaic[:, :, ::-1])
print("done with mosaic")

print("starting TSNE")
dataset = R2V2Dataset(args, "val", transform=StandardVideoTransform(args.input_size, "val"), num_images_to_return=1)

data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=R2V2Dataset.collate_fn,
)

with torch.no_grad():
    torch_devices = args.pytorch_gpu_ids
    device = "cuda:" + str(torch_devices[0])
    model = VinceModel(args)

    model.restore()
    model.eval()
    model.to(device)
    all_images = []
    all_features = []

    for batch in tqdm.tqdm(data_loader, total=NUM_IMAGES_IN_TSNE // args.batch_size):
        batch = process_video_data(batch)

        features = model.get_embeddings(batch)["embeddings"]
        images = to_uint8(batch["data"])
        for image, feature in zip(images, features):
            all_images.append(image)
            all_features.append(pt_util.to_numpy(feature))
        if len(all_images) >= NUM_IMAGES_IN_TSNE:
            break

    del data_loader
    all_features = np.array(all_features)
    all_images = np.array(all_images)
    tsne_images = tsne.tsne_image(all_features, all_images, max_feature_size=50, n_threads=args.num_workers)
    cv2.imwrite("tsne_%s.jpg" % args.title, tsne_images[0][:, :, ::-1])
    print("done")
