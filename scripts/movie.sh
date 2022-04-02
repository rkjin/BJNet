#!/usr/bin/env bash
set -x
DATAPATH="/content/CFNet/datasets/"
CUDA_VISIBLE_DEVICES=0 python movie.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model cfnet --maxdisp 256 \
--loadckpt "/content/drive/MyDrive/logdir/checkpoint_000399.ckpt"