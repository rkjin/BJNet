#!/usr/bin/env bash
set -x
DATAPATH="/content/BJNet/datasets/kitti12_15/"
CUDA_VISIBLE_DEVICES=0 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitticombine.txt --testlist ./filenames/kitticombine_val.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 1 --maxdisp 256 \
    --model fused --logdir ./BJNet/logdir  \
    --test_batch_size 1