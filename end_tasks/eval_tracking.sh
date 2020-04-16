#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_tracking"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python run_end_task_eval.py \
  --base-logdir ${BASE_LOG_LOCATION} \
  --title ${TITLE} \
  --description r18-b-256-q-65536-fsize-64-vid-ibc-4-kinetics \
  --solver EndTaskTrackingSolver \
  --dataset GOT10kDataset \
  --data-path /home/xkcd/datasets/got-10k/ \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 0 \
  --backbone ResNet18SiamFCDilated \
  --input-width 224 \
  --input-height 224 \
  --no-save \
  --disable-dataloader \
  --freeze-feature-extractor
