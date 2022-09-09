from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
from datasets.data_io import get_transform, read_all_lines
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

import cv2 

def test():
    #os.makedirs('./predictions', exist_ok=True)
#    for batch_idx, sample in enumerate(TestImgLoader):
    sample ={}
    batch_idx = 0
    while True:
        batch_idx += 1
        left_img = cv2.imread('/content/drive/MyDrive/data/test_img/left/left.bmp',cv2.IMREAD_COLOR)
        right_img = cv2.imread('/content/drive/MyDrive/data/test_img/right/right.bmp',cv2.IMREAD_COLOR)
        # print(left_img.shape, right_img.shape)
        # image size must be 32 * int 
        processed = get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()
        top_pad = 0 # 
        right_pad = 0 #
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                constant_values=0)

        sample['left'] = torch.Tensor(left_img).unsqueeze(dim =0)
        sample['right'] = torch.Tensor(right_img).unsqueeze(dim =0)     
        sample["top_pad"] = torch.Tensor(np.array([top_pad],dtype=int))# 
        sample["right_pad"] = torch.Tensor(np.array([right_pad],dtype=int)) # 
        sample['left_filename'] = 'movieL'
        sample['left_filename'] = 'movieR'
        # pad images

        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])

        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        os.system("nvidia-smi")
        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
#            print(disp_est, top_pad, right_pad, fn)
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[9:, :-6], dtype=np.float32)
            fn = os.path.join("/content/BJNet/prediction/single"+str(batch_idx)+".png")
#            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
#            print(type(disp_est_uint), disp_est_uint.shape, disp_est_uint.max())
            skimage.io.imsave(fn, disp_est_uint)
        if batch_idx == 1:
          break

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests, pred1_s3_up, pred2_s4 = model(sample['left'].cuda(), sample['right'].cuda())
#    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]

if __name__ == '__main__':
    test()

