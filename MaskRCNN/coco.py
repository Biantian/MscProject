from torchvision import datasets


class CocoDetection(datasets.CocoDetection):
    """
    Modification from torchvision.datasets.CocoDetection. Convert segmentation format
    from RLE to Binary. Convert category_id to index.
    The image is a torch array with shape (c x h x w), channel c=3.
    The target is a torch array with shape (c1 x h x w), channel c1 equals the number
    of objects in the image.
    """

    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, ann_file, transform, target_transform, transforms)

        # transform categories
        self.category_ids = sorted(self.coco.cats.keys())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                image is a torch array of the image.
                target is multiple channel of the annotations.
                Both are FloatTensor.
        """
        img, target_ = super(CocoDetection, self).__getitem__(index)

        # convert the annotations to mask
        labels = []
        masks = []
        areas = []
        boxes = []
        iscrowd = []
        image_id = target_[0]['image_id']
        for ann in target_:
            labels += ann['category_id']
            masks += self.coco.annToMask(ann)
            areas += ann['area']
            boxes += ann['bbox']
            iscrowd += ann['iscrowd']

        # put target elements together
        target = {"labels": labels,
                  "masks": masks,
                  "areas": areas,
                  "boxes": boxes,
                  "image_id": image_id,
                  "iscrowd": iscrowd}

        # this will handle all kinds of transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
