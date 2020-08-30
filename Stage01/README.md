# Stage01

## 1. PyTorch Learning 01

File name: pytorch2.ipynb.

*Using MNIST dataset*

1. Method to load training data.
2. MNIST dataset overview.

## 2. PyTorch Learning 02

File name: pytorch3-nn.ipynb

1. Build MLPs.

   learning rate = 0.001

   epoch  = 10

   | ID   | architecture                  | test error <br/>for <br/>best <br/>validation (%) | time () |
   | ---- | ----------------------------- | ------------------------------------------------- | ------- |
   | 1    | 1024, 512,10                  |                                                   |         |
   | 2    | 1536,1024,512,10              |                                                   |         |
   | 3    | 2048, 1536, 1024, 512,10      |                                                   |         |
   | 4    | 2560, 2048, 1536,1024, 512,10 |                                                   |         |
   | 5    | ？                            |                                                   |         |

   如果结果一直都在变好的话再做提升吧。

2. Train and Test

## 3. PyTorch Learning 03

File name: pytorch5.ipynb

1. Build custom NN

   3 Conv Layer + 3 FCN

2. Result is weird, only predict one result.

3. SGD meet local maximum, not enough data.

File name: pytorch6.ipynb

1. how data augmentation in PyTorch works.

File name: pytorch7.ipynb

1. Loss and accuracy seems acceptable, but test result is very low.

File name: pytorch8.ipynb

1. Fixed bug in last version.
2. add drop out, learning rate decade.

# 4. PyTorch Learning 04

File name: pytorch9-AlexNet-Pre.ipynb

1. Pretrained network AlexNet as backbone.

File name: pytorch10-VGG16.ipynb

1. Pretrained network VGG16 as backbone.