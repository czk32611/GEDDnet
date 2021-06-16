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

(1) A 120$\times$120 face image: *face_img*

(2) Two 80$\times$120 eye images: *left_eye_img* and *right_eye_img*

(3) Pitch and yaw gaze angles in radian: *eye_angle*. **Remember pitch first!!**

(4) An integer to index each subject: *subject_index*.
**When the images of a subject are flipped horizontally, the index changes, i.e., subj_index+total_num_subject**

In dataset['face_img'] in train.py, the shape of the mat should be $N \times 120 \times 120$. The shape of dataset['eye_img'] should be $N \times 80 \times 120$. The shape of dataset['eye_angle'] should be $N \times 2$. The shape of dataset['subject_index'] should be $N \times 1$.

### 3. Online Data Augmentation
During training, PreProcess.py will perform online data augmentatioin, including random horizontal flipping, rotate and cropping. The *face_img* will be cropped from 120$\times$120 to 96$\times$96; the *eye_img* will be cropped from 80$\times$120 to 64$\times$96; The *subject_index* will changes to *subject_index* + *total_num_subject* if the image is flipped horizontally.

### 4. Training and Testing
For training, just simplily run:

    cd code
    python train.py --num_subject *total_num_subject_ignoring_horizontal_flipping*

For inference, run
  
    cd code
    python infer.py

Note that a trained model `data/models` and an example of camera matrix `data/camera_matrix.mat` are provided.

### Bibtex

    @article{chen2020geddnet,
     title={GEDDnet: A Network for Gaze Estimation with Dilation and Decomposition},
     author={Chen, Zhaokang and Shi, Bertram E},
     journal={arXiv preprint arXiv:2001.09284},
     year={2020}
     }
