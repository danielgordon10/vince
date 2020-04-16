#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_imagenet"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --base-logdir ${BASE_LOG_LOCATION} \
  --title ${TITLE} \
  --description r50-moco-v2-b-896-q-65536 \
  --solver EndTaskImagenetSolver \
  --lr-decay-type step \
  --lr-step-schedule 60 80 \
  --epochs 100 \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 0 \
  --backbone ResNet50 \
  --dataset ImagenetDataset \
  --batch-size 256 \
  --image-log-frequency 10000 \
  --save-frequency 2500 \
  --long-save-frequency 25 \
  --num-workers 16 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --base-lr 30.0 \
  --imagenet-data-path /home/xkcd/datasets/imagenet/ \
  --end-task-classifier-num-classes 1000 \
  --freeze-feature-extractor
