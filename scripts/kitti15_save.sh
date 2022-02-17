#!/usr/bin/env bash
set -x
DATAPATH="/content/BJNet/datasets/kitti12_15/Kitti/"
CUDA_VISIBLE_DEVICES=0 python save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model fused --maxdisp 256 \
--loadckpt "/content/drive/MyDrive/logdir/checkpoint_000397.ckpt"