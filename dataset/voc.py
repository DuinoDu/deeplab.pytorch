# Origin: http://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Edited: duinodu

import torch.utils.data as data
import os, sys
import torch
import cv2


class VOCSegmentation(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self.name = 'VOC2007'
        self._annopath = os.path.join(self.root, self.name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(self.root, self.name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, self.name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as fid:
            self.ids = fid.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):

        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self._annopath % img_id)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img, target = self.transform(img, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target 

    def __len__(self):
        return len(self.ids)


def seg_collate_fn(batch):
    """Custom collate_fn for dealing with batches of images.

    Args:
        batch (tuple): A tuple of tensor images and targets

    Returns:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (tensor) batch of targets stacked on their 0 dim
    """
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])
    return torch.stack(images, 0), torch.stack(targets, 0)


def pascal_classes():
    classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4,
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8,
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12,
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16,
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}
    return classes

def pascal_palette():
    #          (  R,   G,   B)
    palette = {(  0,   0,   0) : 0 ,
               (128,   0,   0) : 1 ,
               (  0, 128,   0) : 2 ,
               (128, 128,   0) : 3 ,
               (  0,   0, 128) : 4 ,
               (128,   0, 128) : 5 ,
               (  0, 128, 128) : 6 ,
               (128, 128, 128) : 7 ,
               ( 64,   0,   0) : 8 ,
               (192,   0,   0) : 9 ,
               ( 64, 128,   0) : 10,
               (192, 128,   0) : 11,
               ( 64,   0, 128) : 12,
               (192,   0, 128) : 13,
               ( 64, 128, 128) : 14,
               (192, 128, 128) : 15,
               (  0,  64,   0) : 16,
               (128,  64,   0) : 17,
               (  0, 192,   0) : 18,
               (128, 192,   0) : 19,
               (  0,  64, 128) : 20,
               (224, 224, 192) : 0}
    return palette

def pascal_palette_invert():
    palette = ()
    
    palette_list = pascal_palette().keys()
    for color in palette_list:
        palette += color

    return palette

if __name__ == "__main__":
    ds = VOCSegmentation('/home/dumin/data/VOCdevkit', 'trainval')
    img, target = ds[0]
