# Stage01

## 1. MNIST Hand Writing Dataset

### Check dataset

1. Print digit samples in a line. [pytorch1](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch01.ipynb).
2. Load Dataset and Check dataset composition. [pytorch2](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch2.ipynb).

### Train Models

1. Build a NN with four FC as hidden layers. 
   Trained for 3 epochs, achieved 96.% accuracy on test set. [pytorch3-nn](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch3-nn.ipynb).

2. Print two Learning Rate Decay strategies learning rate line. [pytorch3_lr_decay](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch3_lr_decay.ipynb).

3. Trained 4 NN and compared them. [pytorch3_nn](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch3_nn.ipynb).

   | id   | architecture               | best val err[%] | tst err[%] | training time |
   | ---- | -------------------------- | --------------- | ---------- | ------------- |
   | 1    | 1024,512,10                | 1.351           | 1.39       | 11min53s      |
   | 2    | 1536,1024,512,10*          | 1.331           | 1.38       | 8min46s       |
   | 3    | 2048,1536,1024,512,10      | 1.361           | 1.28       | 19min35s      |
   | 4    | 2560,2048,1536,1024,512,10 | 1.281           | 1.32       | 10min52s      |

   Another file to help train different networks at same time on Colab. [pytorch3_nn_2](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch3_nn_2.ipynb).

## 2. Flower Recognition Dataset

This dataset is on Kaggle: [flower recognition](https://www.kaggle.com/alxmamaev/flowers-recognition).

### Check Dataset

1. Plot dataset sample for each categories, and check dataset composition. [pytorch4](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch4.ipynb).

### First Model

1. Build a ConvNet and tested result.[pytorch5](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch5.ipynb).
   $INPUT\rightarrow[Conv,ReLU,MaxPool]\times3\rightarrow \\ [FC,ReLU]\times2\rightarrow FC\rightarrow OUTPUT$

   Trained 1 epochs, 25.5% accuracy. Bad model.

2.  Test Data Augmentation strategies.   [pytorch6](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch6.ipynb).

3.  Implement data augmentation,  adjust network structure, changed evaluation method, trained on GPU. [pytorch7](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch7.ipynb).
   Trained 6 epochs, according to learning rate curve, train accuracy increased, loss decreased correctly, while validation accuracy is only 25%. Bad Result.

4. Solve the bug in plotting accuracy curve. Use another way to perform network structure. [pytorch8](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch8.ipynb). 

   Trained 40 epochs, training accuracy exceed 90%, validation accuracy is around 70%. The loss curve is a mess, test loss curve is below the train loss. Bad evaluation.

5. Bring in more Regularization method and reduce the capacity of network by delete one FC layer. [pytorch8-1](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch8-1.ipynb).

   *Loss curve bug was not noticed.* 

   $INPUT\rightarrow[Conv,ReLU,MaxPool]\times3\rightarrow \\ FC,ReLU\rightarrow FC\rightarrow OUTPUT$

   60 epochs, train acc: 91.21%, val acc: 69%, time: 33min 57s.

   100 epochs, train acc: 93% , val acc: 69%, time: 59min 7s. [pytorch8-2](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch8-2.ipynb).

   Loss curve shown strong fluctuations, which were misunderstood as learning rate being too big.

6.  Loss curve bug was fixed. Training result can be correctly evaluated till now.  Network is trained on Colab GPU. [pytorch8_1](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch8_1.ipynb).
   10 epochs, time: 3min52s
   train acc: 79.2%, val acc: 70%, 
   train loss: 0.57, val loss: 0.79, 

   Network is underfitting. 

### Optimization and Change the structure

1. Set different 5 different networks. Some of them are overfitting, and the work are focus on avoid overfitting and improve the accuracy on test set. [pytorch8_1_local](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch8_1_local.ipynb). 
   Evaluation method was strongly improved, split dataset into train/val/test set, calculate the precision, recall and f1 score for all categories, draw confusion matrix. Finally, print some correct and incorrect image samples.
   All trained for 40 epochs.

   | net id | test accuracy | time     |
   | ------ | ------------- | -------- |
   | 1      | 0.7191        | 24min45s |
   | 2      | 0.6991        | 16min15s |
   | 3      | 0.7238        | 18min56s |
   | 4      | 0.7052        | 18min37s |
   | 5      | 0.7099        | 16min55s |

2. Seems the limit of ConvNet with 3 Conv Layer on this dataset is 72%. To increase the accuracy we add another Conv Layer. [pytorch8_2_local](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch8_2_local.ipynb).

   | net id | test accuracy | time     |
   | ------ | ------------- | -------- |
   | 6      | 0.7423        | 16min27s |
   | 7      | 0.7454        | 16min25s |

   In 7 loss curve shows no more overfitting, while result is still not satisfying.

   More data are needed, more Conv Layers, Pretraining network on ImageNet or other dataset.

### Pretrained network

1. Reset classifier of AlexNet and train it for [20](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch9-AlexNet-Pre.ipynb) and [50](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch9-AlexNet-Pre-2.ipynb) epochs, result is the same, around 84% accuracy.
2. Reset classifier of ResNet16 and train it for [20](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch10-VGG16.ipynb) and [50](https://github.com/Biantian/MscProject/blob/master/Stage01/pytorch10_VGG16-1.ipynb) epochs, result is the same, around 82% accuracy.

