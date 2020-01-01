#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:12:58 2017

@author: zchenbc
"""
import random
import numpy as np
import cv2
import math
import tensorflow as tf

def _vec22d(input_vec):
    # change 2D angle to 3D vector
    # input_angle: (vertical, horizontal)
    angle = np.stack([np.arcsin(input_vec[:,1]), 
                    np.arctan2(input_vec[:,0],input_vec[:,2])],axis=1)
    return angle


def randomRotate(input_image, input_eye):
    """ rotate the face in x-y plane. change the corresponding eye angle"""
    # num_image = int(round(input_label.shape[0] + sum(input_label)*19))
    input_size = input_image.shape[1:3]
    num_image = input_image.shape[0]
    
    output_image = np.array(input_image, dtype = np.float32)
    output_eye = np.zeros((num_image, 3), dtype=np.float32)
    
    for ii in range(num_image):
        current_image = np.array(input_image[ii], dtype = np.float32)
        angle = (random.randint(0, 300)-150) / 10.
        M = cv2.getRotationMatrix2D((input_size[1]//2.,input_size[0]//2.), angle, 1.)
        current_image = cv2.warpAffine(current_image, M, input_size)
        # transformation matrix for gaze vector
        M_e = np.vstack((M, np.array([[0.,0.,1.]])))
        M_e[0:2,2] = 0
        output_image[ii] = current_image
        output_eye[ii] = np.dot(M_e, input_eye[ii])
        
    output_eye = _vec22d(output_eye)
    return output_image, output_eye


def preprocess_eye_image(image, Augmentation, offset=(5,8), resize_size=(80, 120), pad_size = (96, 150), out_size = (64,96)):
    
    x_processed = tf.stack([image,image,image], axis = 2)
    
    if Augmentation == True:
        
        # input is 80*120, then crop 64*96
        angle = tf.random_uniform([1], minval=-0.087, maxval=0.087)
        x_processed = tf.contrib.image.rotate(x_processed, angle)
        scale = tf.concat([tf.random_uniform([1], minval = 0.85, maxval = 1.15), tf.random_uniform([1], minval = 0.9, maxval = 1.1)],axis=0)
        x_processed = tf.image.resize_images(x_processed, tf.cast(tf.round(scale*resize_size),dtype=tf.int32))
        img_shape = tf.shape(x_processed)[0:2]
        pad = tf.cast(tf.round(0.5*tf.cast(tf.constant(pad_size) - img_shape,dtype=tf.float32)),dtype=tf.int32)
        # pad to 74 * 112
        x_processed = tf.image.pad_to_bounding_box(x_processed, 
                                pad[0],pad[1],pad_size[0], pad_size[1]) 
        
        random_offset_y = tf.random_uniform([1],minval=-offset[0],maxval=offset[0], dtype=tf.int32)
        random_offset_x = tf.random_uniform([1],minval=-offset[1],maxval=offset[1], dtype=tf.int32)
        corner_y = random_offset_y[0] + tf.constant(pad_size[0]//2-out_size[0]//2,dtype=tf.int32)
        corner_x = random_offset_x[0] + tf.constant(pad_size[1]//2-out_size[1]//2,dtype=tf.int32)
        x_processed = tf.image.crop_to_bounding_box(x_processed, 
                                corner_y,corner_x, out_size[0], out_size[1])
        
    else:
        #x_processed = tf.image.resize_images(x_processed, resize_size)
        corner_y = tf.constant(resize_size[0]//2-out_size[0]//2,dtype=tf.int32)
        corner_x = tf.constant(resize_size[1]//2-out_size[1]//2,dtype=tf.int32)
        x_processed = tf.image.crop_to_bounding_box(x_processed, 
                                corner_y, corner_x, out_size[0], out_size[1])
    return x_processed


def pre_process_eye_images(image_list, Augmentation):
    
    images = tf.map_fn(lambda image: preprocess_eye_image(image, Augmentation), image_list)
    return images


def preprocess_face_image(image, Augmentation, offset=(12,12), resize_size=(120, 120), pad_size = (150, 150), out_size = (96,96)):
    
    x_processed = tf.stack([image,image,image], axis = 2)
    
    if Augmentation == True:
        # input is 80*120, then crop 64*96
        scale = tf.random_uniform([2], minval = 0.85, maxval = 1.15)
        x_processed = tf.image.resize_images(x_processed, tf.cast(tf.round(scale*resize_size),dtype=tf.int32))
        img_shape = tf.shape(x_processed)[0:2]
        pad = tf.cast(tf.round(0.5*tf.cast(tf.constant(pad_size) - img_shape,dtype=tf.float32)),dtype=tf.int32)
        # pad to 74 * 112
        x_processed = tf.image.pad_to_bounding_box(x_processed, 
                                pad[0],pad[1],pad_size[0], pad_size[1]) 
        
        random_offset_y = tf.random_uniform([1],minval=-offset[0],maxval=offset[0], dtype=tf.int32)
        random_offset_x = tf.random_uniform([1],minval=-offset[1],maxval=offset[1], dtype=tf.int32)
        corner_y = random_offset_y[0] + tf.constant(pad_size[0]//2-out_size[0]//2,dtype=tf.int32)
        corner_x = random_offset_x[0] + tf.constant(pad_size[1]//2-out_size[1]//2,dtype=tf.int32)
        x_processed = tf.image.crop_to_bounding_box(x_processed, 
                                corner_y,corner_x, out_size[0], out_size[1])
        
    else:
        corner_y = tf.constant(resize_size[0]//2-out_size[0]//2,dtype=tf.int32)
        corner_x = tf.constant(resize_size[1]//2-out_size[1]//2,dtype=tf.int32)
        x_processed = tf.image.crop_to_bounding_box(x_processed, 
                                corner_y, corner_x, out_size[0], out_size[1])
    return x_processed


def pre_process_face_images(image_list, Augmentation):
    
    images = tf.map_fn(lambda image: preprocess_face_image(image, Augmentation), image_list)
    return images


def flip_images(face,left,right, eye, subj, flip_rate, num_subj, block_rate=0.1):   
    face_out = np.array(face,dtype=np.uint8)
    left_out = np.array(left,dtype=np.uint8)
    right_out = np.array(right,dtype=np.uint8)
    eye_out = np.array(eye,dtype=np.float32)
    subj_out = np.array(subj, dtype=np.int_)
    
    batch_size, img_height, img_width = np.shape(face)
    flip_indic = np.random.uniform(0.,1., batch_size)
    for img_id in np.arange(np.shape(face)[0]):
        if flip_indic[img_id] <= flip_rate:
            face_out[img_id] = face[img_id,:,::-1]
            left_out[img_id] = right[img_id,:,::-1]
            right_out[img_id] = left[img_id,:,::-1]
            eye_out[img_id,1] = -eye[img_id,1]
            if subj[img_id, 1] < num_subj:
                subj_out[img_id, 1] += num_subj
            else:
                subj_out[img_id, 1] -= num_subj

    return face_out, left_out, right_out, eye_out, subj_out
