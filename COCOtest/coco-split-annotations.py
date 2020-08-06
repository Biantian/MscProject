# Create annotation file for SubCOCO

from pycocotools.coco import COCO
import json
import os
dsNm = "val2017"
catNms = ['person', 'car', 'bicycle']
root = "E:/Resource/Dataset/COCO"  # addr of COCO
dataDir = os.path.join(root, dsNm)
annDir = os.path.join(root, "annotations/instances_{}.json").format(dsNm)

INFO = {
    "description": "SubCOCO, sub dataset of COCO",
    "url": "https://github.com/Biantian/MscProject/tree/master/COCOtest",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "Biantian",
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": [],
        "images": [],
        "annotations": []
    }

# Initialize coco api
coco = COCO(annDir)

# images
catIds = coco.getCatIds(catNms=catNms)
imgIds = coco.getImgIds(catIds=catIds)
imgs = coco.loadImgs(imgIds)
coco_output['images'] = imgs

# annotations
annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
anns = coco.loadAnns(annIds)
coco_output['annotations'] = anns

# categories
cats = coco.loadCats(catIds)
coco_output['categories'] = cats

# save
with open('{}/SubCOCO/annotation-little.json'.format(root), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)
