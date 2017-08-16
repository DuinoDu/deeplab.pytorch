#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import argparse
from deeplab import build_model 
from dataset import VOCSegmentation, VOCSegAug, seg_collate_fn, evaluate

import os
from tqdm import tqdm
import numpy as np
import pickle

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

parser = argparse.ArgumentParser(description='Deeplab v2 Evaluation')
parser.add_argument('--weight', default='resnet50.pth', type=str, help='trained base model')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--voc_root', default='data/VOCdevkit/', type=str, help='Location of VOC root directory')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size of once evaluation')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

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

# evaluate
def eval():

    print('Loading Dataset...')
    dataset = VOCSegmentation(args.voc_root, 'test', transform=VOCSegAug())
    data_loader = data.DataLoader(dataset, args.batch_size, collate_fn=seg_collate_fn, pin_memory=True)
    t = tqdm(total=len(dataset), unit='batch')
    
    print('Evaluating deeplab on', dataset.name)
    preds, gts = [], []
    
    if os.path.exists('cache/eval.pkl'):
        print('Loading from cache/eval.pkl')
        with open('cache/eval.pkl', 'r') as fid:
            (preds, gts) = pickle.load(fid)
    else:
        for (images, targets) in data_loader:
            t.update(args.batch_size)

            if args.cuda:
                images = Variable(images.cuda())
                targets = Variable(targets.cuda())
            else:
                images = Variable(images)
                targets = Variable(targets)
            
            # infer
            out = net(images)

            #pred = np.squeeze(out.data.max(1)[1].cpu().numpy(), axis=1)
            pred = out.data.max(1)[1].cpu().numpy()
            gt = targets.data.cpu().numpy()

            for gt_, pred_ in zip(gt, pred):
                gts.append(gt_)
                preds.append(pred_)
        with open('cache/eval.pkl', 'w') as fid:
            pickle.dump((preds, gts), fid)
        print('Saving to cache/eval.pkl')

    # compute IOU
    score, class_iou = evaluate(preds, gts, num_classes)

    for k, v in score.items():
        print(k, v)

    for i in range(num_classes):
        print(i, class_iou[i])

if __name__ == "__main__":
    eval()
