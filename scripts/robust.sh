#!/usr/bin/env bash
set -x
DATAPATH="/content/drive/MyDrive/data/kitti12_15"
CUDA_VISIBLE_DEVICES=0 python robust.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitticombine.txt --batch_size 1 --test_batch_size 1 \
    --testlist ./filenames/kitticombine_val.txt --maxdisp 256 \
    --epochs 400 --lr 0.001  --lrepochs "300:10" \
    --loadckpt "/content/drive/MyDrive/data/sceneflow_pretraining.ckpt" \
    --model cfnet --logdir ./CFNet/logdir
