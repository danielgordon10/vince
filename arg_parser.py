import argparse
import multiprocessing
import os

import numpy as np

import constants
import datasets
import solvers
from models.building_blocks import backbone_models
from utils import transforms


def solver_class(class_name):
    if class_name not in solvers.__all__:
        raise argparse.ArgumentTypeError("Invalid solver {}; choices: {}".format(class_name, solvers.__all__))
    return getattr(solvers, class_name)


def dataset_class(class_name):
    if class_name not in datasets.__all__:
        raise argparse.ArgumentTypeError("Invalid dataset {}; choices: {}".format(class_name, datasets.__all__))
    return getattr(datasets, class_name)


def backbone_class(class_name):
    if class_name not in backbone_models.__all__:
        raise argparse.ArgumentTypeError("Invalid backbone {}; choices: {}".format(class_name, backbone_models.__all__))
    return getattr(backbone_models, class_name)


def transform_class(class_name):
    if class_name not in transforms.__all__:
        raise argparse.ArgumentTypeError("Invalid transform {}; choices: {}".format(class_name, transforms.__all__))
    return getattr(transforms, class_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Video Noise Contrastive Estimation training args")

    # Basic args
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--title", type=str, help="Title of method for paths and such.", required=True)
    parser.add_argument("--description", type=str, help="Tag for saving and restoring the model.", required=True)
    parser.add_argument("--num-frames", type=int, help="Number of frames used per video.")
    parser.add_argument(
        "--test-first",
        action="store_true",
        help="True causes val to run before a training epoch. Useful for debugging val without needing to run through a whole epoch.",
    )
    parser.add_argument(
        "--saved-variable-prefix",
        default="",
        type=str,
        help="Used for restoring pytorch models. Prefix to not match for restored variables.",
    )
    parser.add_argument(
        "--new-variable-prefix",
        default="",
        type=str,
        help="Used for restoring pytorch models. Prefix to not match for new variables.",
    )

    # Path args
    parser.add_argument("--base-logdir", metavar="DIR", default=constants.BASE_LOG_DIR, type=str)
    parser.add_argument("--tensorboard-dir", metavar="DIR", default="tensorboard", help="path to tensorboard directory")
    parser.add_argument("--checkpoint-dir", metavar="DIR", help="path to checkpoint directory")
    parser.add_argument("--long-save-checkpoint-dir", metavar="DIR", help="path to long checkpoint directory")

    # Dataset args
    parser.add_argument("--data-path", metavar="DIR", help="path to dataset")
    parser.add_argument("--dataset", help="Dataset to use for training/test.", type=dataset_class)
    parser.add_argument("--transform", default="StandardVideoTransform", help="Transform to use", type=transform_class)

    # Architecture args
    parser.add_argument("--solver", help="Solver to use", type=solver_class)
    parser.add_argument(
        "--backbone", metavar="ARCH", help="backbone to use for feature extraction.", type=backbone_class
    )
    parser.add_argument(
        "--end-task-classifier-num-classes",
        default=0,
        type=int,
        metavar="N",
        help="num classes for the end task classifier to use.",
    )
    parser.add_argument("--use-attention", action="store_true", help="Use attention for averageing the final layers")
    parser.add_argument("--jigsaw", action="store_true", help="Use PIRL Jigsaw method")
    parser.add_argument("--freeze-feature-extractor", action="store_true", help="Do not update the base features.")

    # Loss args
    parser.add_argument(
        "--self-batch-comparison",
        action="store_true",
        help="Whether to compare items from a batch to themselves (vince) for positivies/negatives.",
    )
    parser.add_argument(
        "--inter-batch-comparison",
        action="store_true",
        help="Whether to compare items from a batch to vince network batch for positives/negatives.",
    )

    # VINCE arguments
    parser.add_argument("--vince-queue-size", default=256, type=int, metavar="N", help="num items in the vince queue.")
    parser.add_argument(
        "--vince-embedding-size", default=64, type=int, metavar="N", help="dimensionality of the embedding space."
    )
    parser.add_argument("--vince-momentum", type=float, default=0.999, help="Momentum for vince update")
    parser.add_argument("--vince-temperature", type=float, default=0.07, help="Temperature for NCE in vince")
    parser.add_argument(
        "--vince-self-temperature",
        type=float,
        default=0.03,
        help="Temperature for NCE in vince for nearest neighbor discrimination",
    )
    parser.add_argument("--no-multi-frame", dest="multi_frame", action="store_false",
                        help="Use to disable taking different images from the same video.")

    # Training args
    parser.add_argument("--use-apex", action="store_true", help="Use Nvidia-Apex (automatic mixed precision).")
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="Number of epochs through the data.")
    parser.add_argument("--lr-decay-type", default="cos", choices=["cos", "step"])
    parser.add_argument(
        "--lr-step-schedule",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule for step decay (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--pytorch-gpu-ids",
        type=str,
        default="0",
        help="Comma separated GPU list to use for everything but the feature extractor.",
    )
    parser.add_argument(
        "--feature-extractor-gpu-ids",
        type=str,
        default="0",
        help="Comma separated GPU list to use for the feature extractor.",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        default=multiprocessing.cpu_count(),
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size (default: 256)")
    parser.add_argument("--use-videos", action="store_true", help="Use videos in imagenet training loop")
    parser.add_argument(
        "-e", "--iterations-per-epoch", default=10000, type=int, metavar="N", help="num iterations per pseudo-epoch."
    )
    parser.add_argument("--base-lr", default=0.001, type=float, help="base learning rate ")
    parser.add_argument("--input-width", default=224, type=int, help="Input image width")
    parser.add_argument("--input-height", default=224, type=int, help="Input image height")
    parser.add_argument(
        "--use-imagenet-weights", action="store_true", help="Initialize the model with pretrained imagenet weights"
    )
    parser.add_argument("--no-warmup", dest="use_warmup", action="store_false")

    parser.add_argument("--log-frequency", default=10, type=int, help="Frequency of logging to tensorboard")
    parser.add_argument(
        "--image-log-frequency", default=1000, type=int, help="Frequency of logging images to tensorboard"
    )
    parser.add_argument(
        "--no-save", dest="save", action="store_false", help="Do not save new weights. Useful for debugging."
    )
    parser.add_argument("--no-restore", dest="restore", action="store_false", help="Do not restore previous weights")
    parser.add_argument(
        "--save-frequency", default=5000, type=int, help="Frequency (in iterations) of saving checkpoints"
    )
    parser.add_argument(
        "--long-save-frequency", default=25, type=int, help="Frequency (in epochs) of saving non-deleted checkpoints."
    )
    parser.add_argument(
        "--disable-dataloader", action="store_true", help="Useful during debugging when just running the val."
    )

    # Imagenet Args
    parser.add_argument("--use-imagenet", action="store_true", help="Use imagenet in imagenet training loop")
    parser.add_argument(
        "--imagenet-data-path", type=str, help="Path to imagenet data.", default="/home/xkcd/datasets/imagenet/"
    )

    # Video extraction args
    parser.add_argument(
        "--video-sample-rate", default=5, type=int, help="Number of frames in a video between returned frames"
    )
    parser.add_argument("--max-video-length", type=int, default=512, help="Max number of frames for a video to have.")
    parser.add_argument(
        "--only-use-shots", action="store_true", help="Limits the video caching to single shots within a video."
    )
    parser.add_argument(
        "--max-side-size", default=480, type=int, metavar="N", help="Size to resize the images' longest side."
    )

    args = parser.parse_args()
    args.input_size = (args.input_height, args.input_width)

    assert (not args.inter_batch_comparison) or (
            args.num_frames % 2 == 0
    ), "Must use an even number of frames when not using inter-batch comparison."
    assert (
               not args.self_batch_comparison
           ) or args.inter_batch_comparison, "self-batch-comparison is only used when inter-batch-comparison is on."

    assert args.multi_frame or args.num_frames == 1, "--no-multi-frame only really makes sense with num_frames == 1"

    args.tensorboard_dir = os.path.join(
        args.base_logdir, args.title, args.tensorboard_dir, constants.TIME_STR + "_" + args.description
    )

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.base_logdir, args.title, "checkpoints_" + args.description)

    if args.long_save_checkpoint_dir is None:
        args.long_save_checkpoint_dir = os.path.join(
            args.base_logdir, args.title, "long_checkpoints", constants.TIME_STR + "_" + args.description
        )

    # Reformat GPU IDs
    args.pytorch_gpu_ids = [int(gid) for gid in args.pytorch_gpu_ids.split(",")]
    args.feature_extractor_gpu_ids = [int(gid) for gid in args.feature_extractor_gpu_ids.split(",")]
    all_gpu_ids, gpu_id_map = np.unique(args.pytorch_gpu_ids + args.feature_extractor_gpu_ids, return_inverse=True)
    gpu_id_map = gpu_id_map.tolist()
    all_gpu_ids = ",".join(str(gid) for gid in all_gpu_ids)
    print("Using GPUS", all_gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_ids
    args.pytorch_gpu_ids = gpu_id_map[: len(args.pytorch_gpu_ids)]
    args.feature_extractor_gpu_ids = gpu_id_map[len(args.pytorch_gpu_ids):]

    args.saved_variable_prefix = args.saved_variable_prefix.split(",")
    args.new_variable_prefix = args.new_variable_prefix.split(",")

    print("args")
    print("\n".join([str(key) + ": " + str(val) for key, val in sorted(vars(args).items())]))
    print("-" * 80)

    return args
