# Gaze Decomposition for Appearance-Based Gaze Estimation 
  We proposed gaze decomposition for appearance-based gaze estimation, which decomposes the gaze estimate into the sum of a subject-independent term estimated from the input image by a deep convolutional network and a subject-dependent bias term. 

  During training, both the weights of the deep network and the bias terms are estimated. During testing, if no calibration data is available, we can set the bias term to zero. Otherwise, the bias term can be estimated from images of the subject gazing at different gaze targets. The proposed gaze decompostion method enables low complexity calibraiton, i.e., using calibration data collected when subjects view only one or a few gaze targets and the number of images per gaze target is small.
  
  ![Architecture](https://raw.githubusercontent.com/czk32611/Gaze_Decomposition/master/Figure/Architecture.png)

## Setup
### 1. Prerequisites
Tensorflow

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
    
    @inproceedings{chen2020gaze,
     title={Appearance-based gaze estimation via gaze decomposition and single gaze point calibration},
     author={Chen, Zhaokang and Shi, Bertram E},
     booktitle={Winter Conference on Applications of Computer Vision},
     year={2020},
     organization={IEEE}
     } 
     

