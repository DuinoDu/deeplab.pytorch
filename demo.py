#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
from deeplab import build_model 
from dataset import VOCSegAug
import numpy as np
import cv2
from dataset import pascal_palette

def str2bool(v):
    return v in ['1', 'true', 't' 'yes'] 

parser = argparse.ArgumentParser(description='demo script for sementic segmentation')
parser.add_argument('--weight', default='resnet50.pth', type=str, help='trained base model')
parser.add_argument('--cuda', default=True, type=str2bool, help='use cuda to train model')
parser.add_argument('--image', default=None, type=str, help='demo image path')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else: torch.set_default_tensor_type('torch.FloatTensor')

# parameters
num_classes = 21
gpuID = 0 if args.cuda else -1

# model
net = build_model('resnet50', num_classes, args.weight) 
if args.cuda:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net = net.cuda(gpuID)
print("Create and load model successfully!")

def show_pred(pred, size):
    """Display pred labels, saved in predict.jpg

    :pred (np.ndarray, 1xhxw): TODO
    :size (int): TODO
    :returns: None

    """

    pred = pred.transpose([1,2,0])
    pred_color = np.zeros((pred.shape[0], pred.shape[1], 3))
    palette = pascal_palette()
    for c, i in palette.items():
        m = np.all(pred == np.array(i).reshape(1,1,1), axis = 2)
        pred_color[m] = c
    pred_color = cv2.resize(pred_color, (size[1], size[0]))
    cv2.imwrite('predict.jpg', pred_color) 
    

def mergeImageAndMask(image, mask):
    """Add mask with opacity onto image

    :image (cv2Image, hxwxc)
    :mask: (cv2Image, hxwxc)
    :returns: image with mask 

    """
    opacity = 150./255
    image = cv2.normalize(image.astype(np.float32), 0, 1)
    mask = mask.astype(np.float32)

    target = np.zeros_like(image).astype(np.float32)
    for i in range(3):
        target[:,:,i] = image[:,:,i] * opacity + mask[:,:,i] * (1-opacity)
    return target
 

def demo():
    image = cv2.imread(args.image).astype(np.float32) 
    size = image.shape[:2]
    transform = VOCSegAug()
    image,_ = transform(image)
    
    if args.cuda:
        image = Variable(torch.stack([image], 0).cuda())
    else:
        image = Variable(torch.stack([image], 0))
    
    out = net(image)
    pred = out.data.max(1)[1].cpu().numpy()

    show_pred(pred, size)


if __name__ == "__main__":
    demo()
