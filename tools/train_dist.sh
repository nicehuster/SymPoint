#!/usr/bin/env bash


export PYTHONPATH=./
GPUS=8

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
	--dist configs/svg/svg_pointT.yaml  \
	--exp_name baseline_nclsw_grelu \
	--sync_bn
