from pycocotools.coco import COCO
import os
import shutil

root = "E:/Resource/Dataset/COCO"
dsNm = 'val2017'
input_folder = os.path.join(root, dsNm)
output_folder = os.path.join(root, 'SubCOCO', dsNm)
annDir = os.path.join(root, "SubCOCO/annotation-little.json")

# Initialize coco api
coco = COCO(annDir)

# get imgIds from annotation file
imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)

# move images
for img in imgs:
    file_address = os.path.join(input_folder, img["file_name"])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shutil.copy(file_address, output_folder)
