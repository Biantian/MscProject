# Instance segmentation on Penn-Fudan dataset using Mask RCNN

## Introduction

1. Demonstrate pendn-fudan dataset. (image and mask)
2. Custom PyTorch dataset, to load images and corresponding annotation.
3. Test dataset
4. Defining Mask RCNN model, and reset Mask RCNN predictors.
5. Simple data augmentation (random horizontal flip)
6. Data loader and optimizer (SGD)
7. Train network
8. Plot result.



## Prepare required files

```python
%%shell

# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```

Download the files mentioned to the folder in parallel with the .ipynb file.

由于本地运行的jupyter使用不了上面的代码，所以直接将文件下载到平行的文件夹方便import。

## Note

- `%%shell`以Linux控制台的形式运行代码。

- `cp` copy file 从一个地址到一个地址

- `../`与当前文件夹平行。

