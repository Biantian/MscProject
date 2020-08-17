from pycocotools.coco import COCO
import numpy as np
annFile = 'E:/Resource/Dataset/COCO/SubCOCO/annotations/instances_train2017.json'

# bounding box filter

# initialize coco api
coco = COCO(annFile)
annIds = coco.getAnnIds()
anns = coco.loadAnns(ids=annIds)
deg_ann = {}
for ann in anns:
    bbox = np.array(ann['bbox'])
    if (bbox[:2] < 0).any() or (bbox[2:] <= 0).any():
        deg_ann.update({ann['id']: bbox})
print(deg_ann)