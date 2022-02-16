#!/usr/bin/env bash
set -x
DATAPATH="/content/CFNet/datasets/kitti12_15/"
CUDA_VISIBLE_DEVICES=1 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist --trainlist ./filenames/kitticombine.txt --testlist ./filenames/kitticombine_val.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 1 --maxdisp 256 \
    --model fused --logdir ./CFNet/logdir  \
    --test_batch_size 1