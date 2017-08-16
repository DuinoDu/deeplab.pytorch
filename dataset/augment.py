import numpy as np
import cv2
import random
import torch
from voc import pascal_palette 


class Compose(object):

    """Composes several augmentations together
    Examples:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),   
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        """
        Args:
            transforms (List[Transform]): List of transforms to compose
        """
        self._transforms = transforms

    def __call__(self, image, target=None):
        for t in self._transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)     # h,w,c -> c,h,w
        if target is not None:
            target = torch.from_numpy(target.copy()).type(torch.LongTensor)  # h,w
        return image, target

class ToCV2Image(object):
    def __call(self, image, target=None):
        image = image.cpu().numpy().astype(np.float32)
        if target is not None:
            target = target.cpu().numpy()
        return image, target

class RGB2Label(object):
    def __call__(self, image, target):
        # convert (r,g,b) to (class_label)  
        target_2d = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
        palette = pascal_palette()
        for c, i in palette.items():
            m = np.all(target == np.array(c).reshape(1,1,3), axis=2)
            target_2d[m] = i

        return image, target_2d

class SubstractMean(object):

    def __init__(self, mean):
        self._mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, target=None):
        image -= self._mean
        return image, target

class Resize(object):
    
    def __init__(self, size):
        self._size = size

    def __call__(self, image, target=None):
        image = cv2.resize(image, (self._size, self._size))
        if target is not None:
            target = cv2.resize(target, (self._size, self._size), interpolation = cv2.INTER_NEAREST) 

        return image, target

class Scale(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, image, target):
        target = cv2.resize(target, (self._size, self._size), interpolation = cv2.INTER_NEAREST) 
        return image, target

class RandomMirror(object):
    def __call__(self, image, target=None):
        _, width, _ = image.shape
        if random.randint(0,1):
            image = image[:, ::-1]
            if target is not None:
                target = target[:, ::-1]
        return image, target

class VOCSegAug(object):

    """data augmentation for VOC segmetation dataset"""

    def __init__(self):
        self.mean = (104.008, 116.669, 122.675)
        self.size = 321
        self.scale_size = 41

        self.augment = Compose([
            RGB2Label(),
            Resize(self.size),
            Scale(self.scale_size),
            SubstractMean(self.mean),
            RandomMirror(),
            ToTensor(),
        ]) 

        self.preprocess = Compose([
            Resize(self.size),
            SubstractMean(self.mean),
            ToTensor(),
        ]) 

    def __call__(self, img, target=None):

        img = img.astype(np.float32)

        if target is None:
            return self.preprocess(img)

        else:
            return self.augment(img, target) 
