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
from deeplab import build_model, build_loss 
from dataset import VOCSegmentation, VOCSegAug, seg_collate_fn 
import os

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

parser = argparse.ArgumentParser(description='Deeplab v2 Training')
parser.add_argument('--weight', default='resnet50.pth', type=str, help='pretrained base model')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='starting iter, should be used with resume')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_foler', default='weight/', type=str, help='Location to save checkpoint model')
parser.add_argument('--voc_root', default='data/VOCdevkit/', type=str, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# parameters
num_classes = 21
stepvalues = (80000, 100000, 120000)
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
gamma = args.gamma
gpuID = 0

# model
net = build_model('resnet50', num_classes, args.weight) 
if args.cuda:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net = net.cuda(gpuID)
print("Create and load model successfully!")

# loss and optim
def get_params(net):
    for i in net.parameters():
        if i.requires_grad:
            yield i

criterion = build_loss
optimizer = optim.SGD(get_params(net), lr=lr, momentum=momentum, weight_decay=weight_decay)

# train
def train():

    import tqdm
    t = tqdm.tqdm()
    t.total = args.iterations - args.start_iter

    loss = 0
    epoch = 0

    print('Loading Dataset...')
    dataset = VOCSegmentation(args.voc_root, 'trainval', transform=VOCSegAug())
    epoch_size = len(dataset) // args.batch_size # floor division
    
    print('Training deeplab on', dataset.name)
    batch_iterator = None

    data_loader = data.DataLoader(dataset, args.batch_size, 
                                  shuffle=True, collate_fn=seg_collate_fn, pin_memory=True)

    for iteration in range(args.start_iter+1, args.iterations+1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(data_loader)

        if iteration in stepvalues: # adjust learning rate 
            pass

        images, targets = next(batch_iterator)
    
        if args.cuda:
            images = Variable(images.cuda())
            targets = Variable(targets.cuda())
        else:
            images = Variable(images)
            targets = Variable(images)
        
        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss = criterion(out, targets, gpuID)
        loss.backward()
        optimizer.step()

        t.update()
        if iteration % 10 == 0:
            print('iter '+ repr(iteration) + ' || Loss: %.4f' % (loss.data[0]))

        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), os.path.join(args.save_foler, 'deeplab_' + dataset.name + '_' + str(iteration) + '.pth'))

if __name__ == "__main__":
    train()
