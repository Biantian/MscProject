# Create annotation file for SubCOCO
# Reference: https://github.com/mameng1/split-coco-datasets/blob/master/split_dastaset.py

from pycocotools.coco import COCO
import json
import os
import shutil
from tqdm import tqdm
dsNm = "train2017"
catNms = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
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
imgIds = []
for catId in catIds:
    imgIds = imgIds + coco.getImgIds(catIds=[catId])
imgIds = list(dict.fromkeys(imgIds))  # remove duplicate id
print("Quantity of images: ", len(imgIds))
imgs = coco.loadImgs(imgIds)
coco_output['images'] = imgs

# annotations
annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
print("Quantity of masks: ", len(annIds))
anns = coco.loadAnns(annIds)
coco_output['annotations'] = anns

# categories
cats = coco.loadCats(catIds)
coco_output['categories'] = cats

# save annotations
output_folder = '{}/SubCOCO/annotations'.format(root)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'instances_{}.json'.format(dsNm)), 'w') as output_json_file:
    json.dump(coco_output, output_json_file)
print("Annotation created!")

# copy images
input_folder = os.path.join(root, dsNm)
output_folder = os.path.join(root, 'SubCOCO', dsNm)
print('Copying images\nfrom: {}\nto: {}\n'.format(input_folder, output_folder))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for img in tqdm(imgs):
    file_address = os.path.join(input_folder, img["file_name"])
    shutil.copy(file_address, output_folder)
print("\nImages copied!")
