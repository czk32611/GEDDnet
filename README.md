# GEDDnet: A Network for Gaze Estimation with Dilation and Decomposition

  ![Architecture](https://raw.githubusercontent.com/czk32611/GEDDnet/master/Figure/Architecture.png)

## Dilated Convolution
  We use dilated-convolutions to capture high-level features at high-resolution from eye images. We replace some regular convolutional layers and max-pooling layers of a VGG16 network by dilated-convolutional layers with different dilation rates.

## Gaze Decomposition
  We propose gaze decomposition for appearance-based gaze estimation, which decomposes the gaze estimate into the sum of a subject-independent term estimated from the input image by a deep convolutional network and a subject-dependent bias term.

  During training, both the weights of the deep network and the bias terms are estimated. During testing, if no calibration data is available, we can set the bias term to zero. Otherwise, the bias term can be estimated from images of the subject gazing at different gaze targets. The proposed gaze decompostion method enables low complexity calibraiton, i.e., using calibration data collected when subjects view only one or a few gaze targets and the number of images per gaze target is small.

## Setup
### 1. Prerequisites
Tensorflow == 1.15

python == 3.7

opencv

### 2. Datasets
Preprocess the dataset so that it contains:

(1) A 120*120 face image: *face_img*

(2) Two 80*120 eye images: *left_eye_img* and *right_eye_img*

(3) Pitch and yaw gaze angles in radian: *eye_angle*

### 3. Training and testing
Just simplily run:

    cd code
    python train.py

### Bibtex

    @article{chen2020GEDD,
     title={GEDDnet: A Network for Gaze Estimation with Dilation and Decomposition},
     author={Chen, Zhaokang and Shi, Bertram E},
     }
