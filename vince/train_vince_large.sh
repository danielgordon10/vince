#!/usr/bin/env bash

ulimit -n 99999

TITLE="vince"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --title ${TITLE} \
  --base-logdir ${BASE_LOG_LOCATION} \
  --description r50-b-896-q-65536 \
  --solver VinceSolver \
  --backbone ResNet50 \
  --dataset R2V2Dataset \
  --transform SimCLRTransform \
  --num-workers 40 \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 1,2,3,4,5,6,7 \
  --batch-size 896 \
  --base-lr 0.105 \
  --vince-embedding-size 128 \
  --vince-queue-size 65536 \
  --vince-momentum 0.999 \
  --vince-temperature 0.2 \
  --epochs 200 \
  --save-frequency 1430 \
  --iterations-per-epoch 1430 \
  --image-log-frequency 1430 \
  --long-save-frequency 10 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --use-videos \
  --num-frames 4 \
  --inter-batch-comparison \
  --self-batch-comparison \
  --imagenet-data-path /usr/lusers/xkcd/cse/datasets/imagenet \
  --data-path /usr/lusers/xkcd/cse/datasets/r2v2_large_with_ids/
