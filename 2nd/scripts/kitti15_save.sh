#!/usr/bin/env bash
set -x
DATAPATH="/content/drive/MyDrive/data/kitti12_15/Kitti/"
CUDA_VISIBLE_DEVICES=0 python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model cfnet_modified --maxdisp 256 \
--loadckpt "/content/drive/MyDrive/data/finetuning_model"