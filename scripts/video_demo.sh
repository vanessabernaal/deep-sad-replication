#!/usr/bin/env bash
set -e

mkdir -p results/video-demo

python src/main.py \
  mnist \
  mnist_LeNet \
  results/video-demo \
  data \
  --device cpu \
  --normal_class 0 \
  --known_outlier_class 1 \
  --n_known_outlier_classes 1 \
  --ratio_known_outlier 0.05 \
  --ratio_pollution 0.0 \
  --seed 1 \
  --n_epochs 1 \
  --ae_n_epochs 1 \
  --batch_size 128 \
  --ae_batch_size 128