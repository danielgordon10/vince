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
  --description r18-b-256-q-65536 \
  --solver VinceSolver \
  --backbone ResNet50 \
  --dataset R2V2Dataset \
  --transform StandardVideoTransform \
  --num-workers 4 \
  --pytorch-gpu-ids 6 \
  --feature-extractor-gpu-ids 6 \
  --batch-size 64 \
  --base-lr 0.03 \
  --vince-queue-size 256 \
  --vince-momentum 0.999 \
  --vince-temperature 0.07 \
  --save-frequency 100 \
  --iterations-per-epoch 100 \
  --image-log-frequency 100 \
  --long-save-frequency 10 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --num-frames 4 \
  --no-save \
  --no-restore \
  --inter-batch-comparison \
  --self-batch-comparison \
  --imagenet-data-path /home/xkcd/datasets/imagenet \
  --data-path /home/xkcd/datasets/r2v2_large_with_ids/ \
  --use-videos \
  --use-apex
#--test-first \
#--debug \
#--imagenet-data-path /usr/lusers/xkcd/datasets/imagenet \
#--data-path /gscratch/scrubbed/xkcd/datasets/r2v2_large_with_ids/ \
