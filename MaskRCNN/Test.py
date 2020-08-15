from pycocotools.coco import COCO
import os
import torch
from torchvision.transforms import transforms
from .engine import train_one_epoch, evaluate
from .utils import *
from .transforms import *


root = 'E:/Resource/Dataset/COCO/SubCOCO'
annDir = os.path.join(root, 'annotations/instances_{}.json')
# coco = COCO(annDir.format('train2017'))

def get_transform(train):
    transforms = []
    transforms.append(T .ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)