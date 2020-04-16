import os
import random

import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import tqdm
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from dg_util.python_utils.persistent_dataloader import PersistentDataLoader
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader

import arg_parser
from datasets.r2v2_dataset import R2V2Dataset
from models.vince_model import VinceModel
from utils.transforms import RepeatedImagenetTransform
from utils.transforms import StandardVideoTransform
from utils.util_functions import to_uint8

"""
Example run command
python visualizations/view_nearest_neighbors.py \
    --title sample_mosaic \
    --description none \
    --checkpoint-dir logs/moco/MocoImagenetModel/checkpoints_r18-b-256-q-65536-fsize-64-vid-ibc-4-no-self/ \
    --data-path /home/xkcd/datasets/r2v2_large_with_ids/ \
    --num-workers 80 --backbone ResNet18 --pytorch-gpu-ids 0 --feature-extractor-gpu-ids 0 \
    -b 512 \
"""

NUM_QUERIES = 100
NUM_NEIGHBORS = 10
NUM_TO_COMPARE = 50000

data_subset = "val"
args = arg_parser.parse_args()


def get_data_item(data):
    if isinstance(data, dict):
        data = data["data"]
        data = data.squeeze(1)
    elif isinstance(data, list) or isinstance(data, tuple):
        data, label = data
        data = data.squeeze(1)
    else:
        raise NotImplementedError
    return data


def dataset_nn(model, data_loader):
    with torch.no_grad():
        num_to_compare = min(int(NUM_TO_COMPARE / args.batch_size + 1) * args.batch_size, len(data_loader.dataset))

        # Get features
        image_array = np.zeros((num_to_compare, args.input_height, args.input_width, 3), dtype=np.uint8)
        features_array = None

        data_ind = 0
        pbar = tqdm.tqdm(total=num_to_compare)
        for data in data_loader:
            data = get_data_item(data)
            data_size = data.shape[0]
            data = data.to(model.device)
            output = model.get_embeddings({"data": data, "batch_type": ("images", len(data))})
            features = output["extracted_features"]

            if features_array is None:
                feature_size = features.shape[1]
                features_array = torch.zeros((num_to_compare, feature_size), dtype=torch.float32, device=model.device)
            features_array[data_ind: data_ind + data_size] = features
            data = to_uint8(data)
            image_array[data_ind: min(num_to_compare, data_ind + data_size)] = data
            data_ind += data_size
            pbar.update(data_size)
            if data_ind >= num_to_compare:
                break
        pbar.close()
        if features_array.shape[1] != 64:
            features_array_new = pt_util.to_numpy(features_array)
            pca = PCA(n_components=64)
            features_array_new = pca.fit_transform(features_array_new)
            features_array_new = pt_util.from_numpy(features_array_new).to(features_array.device)
            features_array = features_array_new
            features_array = torch.nn.functional.normalize(features_array, dim=-1)
        return features_array, image_array


def draw_nns(source_features, source_images, source_name, target_features=None, target_images=None, target_name=None):
    skip_first = False
    if target_features is None:
        target_features = source_features
        target_images = source_images
        target_name = source_name
        skip_first = True

    num_to_compare = target_features.shape[0]
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    rand_selection = np.sort(np.random.choice(source_features.shape[0], NUM_QUERIES, replace=False))

    query_features = source_features[rand_selection]

    dists = torch.mm(query_features, target_features.T)
    val, neighbors = torch.topk(dists, k=(NUM_NEIGHBORS + int(skip_first)), dim=1, sorted=True, largest=True)
    if skip_first:
        neighbors = neighbors[:, 1:]

    neighbors = target_images[pt_util.to_numpy(neighbors)]
    os.makedirs(
        os.path.join(args.checkpoint_dir, "neighbors_from_%s_to_%s" % (source_name, target_name)), exist_ok=True
    )

    # Get images
    for ii in tqdm.tqdm(range(neighbors.shape[0])):
        images = []
        image = source_images[rand_selection[ii]].copy()
        image = np.pad(image, ((10, 10), (10, 10), (0, 0)), "constant")
        images.append(image)
        for jj in range(neighbors.shape[1]):
            image = neighbors[ii, jj].copy()
            images.append(image)

        subplot = drawing.subplot(images, 1, neighbors.shape[1] + 1, args.input_width, args.input_height, border=5)
        cv2.imwrite(
            os.path.join(
                args.checkpoint_dir,
                "neighbors_from_%s_to_%s" % (source_name, target_name),
                "bsize_%06d_%03d.jpg" % (num_to_compare, ii),
            ),
            subplot[:, :, ::-1],
        )


def main():
    with torch.no_grad():
        torch_devices = args.pytorch_gpu_ids
        device = "cuda:" + str(torch_devices[0])
        model = VinceModel(args)

        model.restore()
        model.eval()
        model.to(device)
        yt_dataset = R2V2Dataset(
            args, "val", transform=StandardVideoTransform(args.input_size, "val"), num_images_to_return=1
        )
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        data_loader = PersistentDataLoader(
            yt_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=R2V2Dataset.collate_fn,
            worker_init_fn=R2V2Dataset.worker_init_fn,
        )

        yt_features, yt_images = dataset_nn(model, data_loader)
        del data_loader

        draw_nns(yt_features, yt_images, "youtube")

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        valdir = os.path.join(args.imagenet_data_path, data_subset)
        transform = RepeatedImagenetTransform(args.input_height, data_subset="val", repeats=1)
        imagenet_dataset = datasets.ImageFolder(valdir, transform)
        data_loader = DataLoader(
            imagenet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )

        imagenet_features, imagenet_images = dataset_nn(model, data_loader)
        del data_loader

        draw_nns(imagenet_features, imagenet_images, "imagenet")
        draw_nns(imagenet_features, imagenet_images, "imagenet", yt_features, yt_images, "youtube")
        draw_nns(yt_features, yt_images, "youtube", imagenet_features, imagenet_images, "imagenet")


if __name__ == "__main__":
    main()
