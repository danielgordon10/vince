#!/usr/bin/env bash
# Reproducing results from https://arxiv.org/pdf/2003.04297.pdf
ulimit -n 99999

TITLE="moco_v2"
BASE_LOG_LOCATION="logs"
LOG_LOCATION=${BASE_LOG_LOCATION}"/"${TITLE}

mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python solver_runner.py \
  --title ${TITLE} \
  --base-logdir ${BASE_LOG_LOCATION} \
  --description r50-moco-v2-b-256-e-128-q-65536 \
  --solver VinceSolver \
  --backbone ResNet50 \
  --transform MoCoV2ImagenetTransform \
  --num-workers 40 \
  --pytorch-gpu-ids 0 \
  --feature-extractor-gpu-ids 1,2,3,4,5,6,7 \
  --batch-size 256 \
  --base-lr 0.03 \
  --vince-embedding-size 128 \
  --vince-queue-size 65536 \
  --vince-momentum 0.999 \
  --vince-temperature 0.2 \
  --epochs 200 \
  --lr-decay-type cos \
  --save-frequency 5005 \
  --iterations-per-epoch 5005 \
  --image-log-frequency 5005 \
  --long-save-frequency 10 \
  --log-frequency 10 \
  --input-width 224 \
  --input-height 224 \
  --num-frames 1 \
  --imagenet-data-path /usr/lusers/xkcd/cse/datasets/imagenet \
  --use-imagenet
#--use-apex \
#--data-path /gscratch/scrubbed/xkcd/datasets/r2v2_large_with_ids/ \
#--use-videos \
