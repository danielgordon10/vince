#!/usr/bin/env bash

ulimit -n 99999

TITLE="end_task_sun_scene"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --solver EndTaskSunSceneSolver \
  --lr-decay-type step \
  --lr-step-schedule 500 1000 1500 \
  --epochs 2000 \
  --dataset SunSceneDataset \
  --data-path /home/xkcd/datasets/sun_scene_data/ \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 0 \
  --base-logdir ${BASE_LOG_LOCATION} \
  --backbone ResNet18 \
  --batch-size 16 \
  --title ${TITLE} \
  --iterations-per-epoch 5000 \
  --image-log-frequency 5000 \
  --save-frequency 5000 \
  --long-save-frequency 25 \
  --num-workers 10 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --base-lr 0.03 \
  --no-save \
  --description r50-moco-v2-b-896-q-65536 \
  --end-task-classifier-num-classes 397 \
  --freeze-feature-extractor \
  --test-first \
  --debug
