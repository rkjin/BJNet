#!/usr/bin/env bash
set -x
DATAPATH="/content/CFNet/datasets/kitti12_15/Kitti/"
CUDA_VISIBLE_DEVICES=0 python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model cfnet --maxdisp 256 \
--loadckpt "/content/CFNet/datasets/finetuning_model"