#!/usr/bin/env bash
set -x
DATAPATH="/content/drive/MyDrive/data/kitti12_15/Kitti/"
CUDA_VISIBLE_DEVICES=0 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitticombine.txt --testlist ./filenames/kitticombine_val.txt \
    --epochs 400 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 1 --maxdisp 256 \
    --model cfnet_modified --logdir /content/drive/MyDrive/logdir \
    --test_batch_size 1 --resume 