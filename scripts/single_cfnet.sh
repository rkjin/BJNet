#!/usr/bin/env bash
set -x
DATAPATH="/content/BJNet/datasets/"
CUDA_VISIBLE_DEVICES=0 python single.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model cfnet --maxdisp 256 \
--loadckpt "/content/drive/MyDrive/data/finetuning_model"

# cfnet 
# single picture 