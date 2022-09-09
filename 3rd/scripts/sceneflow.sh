#!/usr/bin/env bash
set -x
DATAPATH="/content/drive/MyDrive/data/kitti12_15/"
CUDA_VISIBLE_DEVICES=0 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitticombine.txt --testlist ./filenames/kitticombine_val.txt \
    --epochs 400 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 1 --maxdisp 256 \
    --model cfnet_modified --logdir /content/drive/MyDrive/logdir \
    --test_batch_size 1  

# original
# #!/usr/bin/env bash
# set -x
# DATAPATH="/home2/dataset/scene_flow/"
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset sceneflow \
#     --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#     --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 1 --maxdisp 256 \
#     --model cfnet --logdir ./checkpoints/sceneflow/uniform_sample_d256  \
#     --test_batch_size 1