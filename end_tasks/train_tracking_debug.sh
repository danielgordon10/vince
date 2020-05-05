#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_tracking"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --solver EndTaskTrackingSolver \
  --lr-decay-type step \
  --lr-step-schedule 10 20 30 \
  --epochs 40 \
  --dataset GOT10kDataset \
  --data-path /home/xkcd/datasets/got-10k-small/ \
  --pytorch-gpu-ids 6 \
  --feature-extractor-gpu-ids 6 \
  --base-logdir ${BASE_LOG_LOCATION} \
  --backbone ResNet18SiamFCDilated \
  --batch-size 32 \
  --title ${TITLE} \
  --iterations-per-epoch 5000 \
  --image-log-frequency 5000 \
  --save-frequency 5000 \
  --long-save-frequency 25 \
  --num-workers 4 \
  --log-frequency 10 \
  --base-lr 0.001 \
  --no-save \
  --description r18-b-256-q-65536-fsize-64-vid-ibc-4-no-self-unfrozen-backbone \
  --debug \
  --freeze-feature-extractor
