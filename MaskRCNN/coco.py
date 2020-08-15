import torch
import numpy as np
from torchvision import datasets
import numpy as np  # (pip install numpy)
# from skimage import measure  # (pip install scikit-image)
# from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)

DEBUG = True


class CocoSubset(datasets.CocoDetection):
    """
    Modification from torchvision.datasets.CocoDetection. Convert segmentation format
    from RLE to Binary. Convert category_id to index.
    The image is a torch array with shape (c x h x w), channel c=3.
    The target is a torch array with shape (c1 x h x w), channel c1 equals the number
    of objects in the image.
    """

    def __init__(self, root, ann_file, both_transform=None, target_transform=None, img_transform=None):
        super(CocoSubset, self).__init__(root, ann_file, None, None)
        self.both_transform = both_transform
        self.target_transform = target_transform
        self.img_transform = img_transform
        # get the categories
        self.category_ids = sorted(self.coco.cats.keys())
        self.ncategories = len(self.category_ids)
        self.cat2idx = {self.category_ids[i]: i for i in range(self.ncategories)}


    # def create_sub_mask_annotation(self, sub_mask):
    #     # Find contours (boundary lines) around each sub-mask
    #     # Note: there could be multiple contours if the object
    #     # is partially occluded. (E.g. an elephant behind a tree)
    #     contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    #
    #     segmentations = []
    #     polygons = []
    #     for contour in contours:
    #         # Flip from (row, col) representation to (x, y)
    #         # and subtract the padding pixel
    #         for i in range(len(contour)):
    #             row, col = contour[i]
    #             contour[i] = (col - 1, row - 1)
    #
    #         # Make a polygon and simplify it
    #         poly = Polygon(contour)
    #         poly = poly.simplify(1.0, preserve_topology=False)
    #         polygons.append(poly)
    #         segmentation = np.array(poly.exterior.coords).ravel().tolist()
    #         segmentations.append(segmentation)
    #
    #     # Combine the polygons to calculate the bounding box and area
    #     multi_poly = MultiPolygon(polygons)
    #     x, y, max_x, max_y = multi_poly.bounds
    #     width = max_x - x
    #     height = max_y - y
    #     bbox = (x, y, width, height)
    #     area = multi_poly.area
    #
    #     annotation = {
    #         'segmentation': segmentations,
    #         'bbox': bbox,
    #         'area': area
    #     }
    #
    #     return annotation


    def update_target(self, target_):
        boxes = []
        areas = []
        masks = target_['masks'].numpy()
        for idx in range(len(masks)):
            pos = np.where(masks[idx])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append(masks[idx].sum())
        target_['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target_['areas'] = torch.as_tensor(areas, dtype=torch.float32)
        return target_

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
        img, target = super(CocoSubset, self).__getitem__(index)
        # convert the annotations to mask
        labels = []
        masks = []
        areas = []
        boxes = []
        iscrowd = []
        image_id = target[0]['image_id']
        for ann in target:
            labels.append(self.cat2idx[ann['category_id']])
            masks.append(self.coco.annToMask(ann))
            areas.append(ann['area'])
            boxes.append(ann['bbox'])
            iscrowd.append(ann['iscrowd'])
        # transform boxes from [x, y, wight, height] to [xmin, ymin, xmax, ymax]
        ann = np.array(boxes)
        boxes = np.append(ann[:, :2], np.transpose([ann[:, 0] + ann[:, 2]]), axis=1)
        boxes = np.append(boxes,  np.transpose([ann[:, 1] + ann[:, 3]]), axis=1)
        # put target elements together
        target_ = {"labels": torch.as_tensor(labels, dtype=torch.int64),
                   "masks": torch.as_tensor(masks, dtype=torch.uint8),
                   "areas": torch.as_tensor(areas, dtype=torch.float32),
                   "boxes": torch.from_numpy(boxes).float(),
                   "image_id": torch.tensor([image_id]),
                   "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)}

        # this will handle all kinds of transforms
        if self.both_transform is not None:
            img, target_['masks'] = self.both_transform(img, target_['masks'])
            self.update_target(target_)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target_['masks'] = self.target_transform(target_['masks'])
            self.update_target(target_)
        return img, target_


if DEBUG:
    import os

    root = 'E:/Resource/Dataset/COCO/SubCOCO'
    annDir = os.path.join(root, 'annotations/instances_{}.json')
    coco_train = CocoSubset(os.path.join(root, 'train2017'),
                            annDir.format('train2017'))

    coco_val = CocoSubset(os.path.join(root, 'val2017'),
                          annDir.format('val2017'))
    print('Amount of train images:')
    print(coco_train)
    print('Amount of validation images:')
    print(coco_val)
    print(len(coco_train))
    print(type(coco_train))
    for img, target in coco_train:
        print(img, target)
        break
