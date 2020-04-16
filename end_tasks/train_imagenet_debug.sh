#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_imagenet"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --solver EndTaskImagenetSolver \
  --pytorch-gpu-ids 7 \
  --feature-extractor-gpu-ids 7 \
  --base-logdir ${BASE_LOG_LOCATION} \
  --backbone ResNet50 \
  --dataset ImagenetDataset \
  --batch-size 16 \
  --title ${TITLE} \
  --image-log-frequency 5000 \
  --save-frequency 5000 \
  --long-save-frequency 10 \
  --num-workers 10 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --base-lr 0.03 \
  --no-save \
  --description r50-moco-v2-b-896-q-65536 \
  --imagenet-data-path /home/xkcd/datasets/imagenet/ \
  --end-task-classifier-num-classes 1000 \
  --freeze-feature-extractor \
  --debug \
  --use-apex
