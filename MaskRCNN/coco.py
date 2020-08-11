import os
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

class CocoDetection(datasets.CocoDetection):
    """
    Modification from torchvision.datasets.CocoDetection. Convert segmentation format
    from RLE to Binary. Convert category_id to index.
    The image is a torch array with shape (c x h x w), channel c=3.
    The target is a torch array with shape (c1 x h x w), channel c1 equals the number
    of objects in the image.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)

        # transform categories
        self.category_ids = sorted(self.coco.cats.keys())

    def __getitem__(self, index):
        """
        Args:
        Returns:
        """


    def __len__(self):
        return len()
