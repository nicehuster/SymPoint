#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=8
workdir=./work_dirs/svg/svg_pointT/baseline_nclsw_grelu
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/test.py \
	 $workdir/svg_pointT.yaml  $workdir/best.pth --dist
