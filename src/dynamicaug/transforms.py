import math
import numbers
import sys
import warnings

import torch
from PIL import Image

from . import functional as F
from . import _utils as utils


class DynamicCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, target):
        for t in self.transforms:
            if hasattr(t, 'forward'):
                if utils.get_target(t) and utils.get_input(t):
                    img, target = t(img, target)
                elif utils.get_target(t) and not utils.get_input(t):
                    target = t(target)
                else:
                    img = t(img)
            else:
                img = t(img)
        return img, target            


def DynamicDatasetWrapper(dataset):
    """
    Transform the given dataset's __getitem__ method into DynamicAugment.
    
    Please note that you'll might have to adjust the first line
    'img, target = self.data[index], self.targets[index]'
    to match your dataset's notation of input and output.
    This Wrapper is an example for torchvision.datasets.CIFAR10,
    in which the input is notated as 'data' and the output as 'targets'.
    
    Recommend NOT TO USE THIS WRAPPER, 
    but to just copy the following simple DynamicAugment code into your dataset:
    
        if self.transform is not None:
            img, target = self.transform(img, target)

    Args:
        dataset (class): A typical PyTorch-style dataset which has __init__ method 
            to call data and transform flows

    Returns:
        class: A DynamicAugment applicatable dataset class
    """
    class DynamicDatasetWrapper(dataset):
        def __getitem__(self, index:int) -> tuple[any, any]:
            img, target = self.data[index], self.targets[index]
            
            img = Image.fromarray(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            if self.transform is not None:
                img, target = self.transform(img, target)
            
            return img, target
    return DynamicDatasetWrapper


class CutOut(torch.nn.Module):
    def __init__(self, p=.2, fillcolor=0):
        super().__init__()
        self.p = p
        self.fillcolor = fillcolor
    
    def forward(self, img):
        return F.cutout(img, self.p, self.fillcolor)


class DynamicCutOut(torch.nn.Module):
    def __init__(self, p=.2, fillcolor=0, mode=1):
        super().__init__()
        self.p = p
        self.fillcolor = fillcolor
        self.mode = mode
    
    def forward(self, img, target):
        return F.d_cutout(img, target, self.p, self.fillcolor, self.mode)
    

class GridMask(torch.nn.Module):
    def __init__(self, d, r, fillcolor=0, rotate=True):
        super().__init__()
        self.d = d
        self.r = r
        self.fillcolor = fillcolor
        self.rotate = rotate
    
    def forward(self, img):
        return F.gridmask(img, self.d, self.r, self.fillcolor, self.rotate)


class DynamicGridMask(torch.nn.Module):
    def __init__(self, d, r, mode=1, fillcolor=0):
        super().__init__()
        self.d = d
        self.r = r
        self.mode = mode
        self.fillcolor = fillcolor
    
    def forward(self, img, target):
        return F.d_gridmask(img, target, self.mode, self.d, self.r, self.fillcolor)


class PatchGridMask(torch.nn.Module):
    def __init__(self, p, n, r, fillcolor=0):
        super().__init__()
        self.p = p
        self.n = n
        self.r = r
        self.fillcolor = fillcolor
    
    def forward(self, img):
        return F.patch_gridmask(img, self.p, self.n, self.r, self.fillcolor)


class OneHot(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, target):
        return torch.nn.functional.one_hot(torch.tensor(target), self.num_classes).float()
