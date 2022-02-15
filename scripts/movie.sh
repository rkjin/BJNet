#!/usr/bin/env bash
set -x
DATAPATH="/content/CFNet/datasets/movie/"
CUDA_VISIBLE_DEVICES=0 python movie.py --datapath $DATAPATH --testlist ./filenames/movie_data.txt --model cfnet --maxdisp 256 \
--loadckpt "/content/CFNet/datasets/finetuning_model"