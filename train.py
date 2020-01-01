# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import os.path as osp
import tensorflow as tf
import numpy as np
import random
from tf_utils import conv2d, dilated2d, max_pool_2x2, weight_variable, bias_variable, dense_to_one_hot
from PreProcess import randomRotate, pre_process_eye_images, pre_process_face_images, flip_images
from dilatedNet import dilatedNet
import matplotlib.pyplot as plt
import math
import scipy.io as spio

FLAGS = None

def _2d2vec(input_angle):
    # change 2D angle to 3D vector
    # input_angle: (vertical, horizontal)
    vec = np.stack([np.sin(input_angle[:,1])*np.cos(input_angle[:,0]), 
                    np.sin(input_angle[:,0]), 
                    np.cos(input_angle[:,1])*np.cos(input_angle[:,0])],axis=1)
    return vec

def _vec22d(input_vec):
    # change 2D angle to 3D vector
    # input_angle: (vertical, horizontal)
    angle = np.stack([np.arcsin(input_vec[:,1]), 
                    np.arctan2(input_vec[:,0],input_vec[:,2])],axis=1)
    return angle

def _angle2error(vec1, vec2):
    # calculate the angular difference between vec1 and vec2
    tmp = np.sum(vec1*vec2, axis = 1)
    tmp = np.maximum(tmp, -1.)
    tmp = np.minimum(tmp, 1.)
    angle = np.arccos(tmp) * 180. / np.pi
    
    return angle

def creatIter(index, batch_size, isShuffle = False):
    dataset = tf.data.Dataset.from_tensor_slices(index)
    if isShuffle == True:
        dataset = dataset.shuffle(buffer_size = index.shape[0])
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    
    return iterator, next_element

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(_):
    
    input_size = (80, 120)
    num_subj = 15
    
    mu = np.array([123.68, 116.779, 103.939], dtype = np.float32).reshape(
        (1, 1, 3))
    
    keep_prob = tf.placeholder(tf.float32)
    is_train = tf.placeholder(tf.bool) 
    is_aug = tf.placeholder(tf.bool)
    
    # input image
    x_f = tf.placeholder(tf.float32, [None, input_size[1], input_size[1]])
    x_l = tf.placeholder(tf.float32, [None, input_size[0], input_size[1]])
    x_r = tf.placeholder(tf.float32, [None, input_size[0], input_size[1]])
    
    
    y_ = tf.placeholder(tf.float32, [None, 2])  # output label
    subj_id = tf.placeholder(tf.float32, [None, 2*num_subj])  # subject index
    
    # online data augmentation
    f_processed = tf.where(is_aug, pre_process_face_images(x_f, True), pre_process_face_images(x_f, False))
    face = f_processed - mu
    l_processed = tf.where(is_aug, pre_process_eye_images(x_l, True), pre_process_eye_images(x_l, False))
    left_eye = l_processed - mu
    r_processed = tf.where(is_aug, pre_process_eye_images(x_r, True), pre_process_eye_images(x_r, False))
    right_eye = r_processed - mu
    
    g_hat, t_hat, bias_W_fc, l2_loss = dilatedNet(face, left_eye, right_eye, 
                                                  keep_prob, is_train, subj_id, 
                                                  vgg_path=FLAGS.vgg_dir)
    
    est_loss = tf.reduce_sum((g_hat - y_ * 10.)**2, axis = 1, keep_dims=True)
        
        
    total_loss = (tf.reduce_mean(est_loss) + 1e-3 * l2_loss + 
                  tf.abs(tf.reduce_mean(bias_W_fc[:num_subj,0]))+tf.abs(tf.reduce_mean(bias_W_fc[num_subj:,0]))+
                  tf.abs(tf.reduce_mean(bias_W_fc[:num_subj,1]))+tf.abs(tf.reduce_mean(bias_W_fc[num_subj:,1])))
    
    cls1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "face") + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "eye") + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "combine") + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "bias")
                
    vgg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "transfer")

    # optimizer
    with tf.variable_scope("LearnRate"):
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.train.exponential_decay(2e-3, global_step, 8000, 0.1, 
                                               staircase = True)
    
    opt1 = tf.train.AdamOptimizer(learning_rate)
    opt2 = tf.train.AdamOptimizer(1e-1*learning_rate)
    
    update_ops1 = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "face") + \
                  tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "eye") + \
                  tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = "combine")
                  
    with tf.control_dependencies(update_ops1):
        optimizer1 = opt1.minimize(total_loss, var_list = cls1_vars, global_step = global_step)
        optimizer2 = opt2.minimize(total_loss, var_list = vgg_vars)
        

    batch_size = 100;
    train_batch_size = 64
    #batch_subj64[:,-1] = 1
    batch_subj100 = np.zeros((batch_size, 2*num_subj))
    
    saver = tf.train.Saver(max_to_keep = 100)
    
    # load data
    for fold in range(1,16):
        dataset = spio.loadmat(FLAGS.data_dir+str(fold)+'train.mat')
        face_train = dataset['face_img']
        left_train = dataset['left_eye_img']
        right_train = dataset['right_eye_img']
        eye_train = dataset['eye_angle']
        train_index = np.arange(eye_train.shape[0])
        vec_train = _2d2vec(eye_train)
        # generate subject index
        subj_train = np.array([]).reshape((0,1))
        subj_train2 = np.array([]).reshape((0,1))
        for ii in range(eye_train.shape[0] // 3000):
            subj_train = np.vstack([subj_train, ii*np.ones((3000, 1))])
            subj_train2 = np.vstack([subj_train2, ii*np.ones((1500, 1))])
            subj_train2 = np.vstack([subj_train2, (ii+num_subj)*np.ones((1500, 1))])
        subj_train = np.concatenate([subj_train, subj_train2], axis=1)
        
        dataset = spio.loadmat(FLAGS.data_dir+str(fold)+'test.mat')
        face_test = dataset['face_img']
        left_test = dataset['left_eye_img']
        right_test = dataset['right_eye_img']
        eye_test = dataset['eye_angle']
        vec_test = _2d2vec(eye_test)
        
        del dataset
        train_index = np.arange(eye_train.shape[0])
        
        train_iter, train_element = creatIter(train_index, train_batch_size, isShuffle = True)
        test_iter, test_element = creatIter(np.arange(eye_test.shape[0]), batch_size)
        sample_size = (eye_train.shape[0], eye_test.shape[0])
        
        if ~osp.exists(str(fold)):
            cmd = 'mkdir ' + str(fold)
            os.system(cmd)
            
        model_path = osp.join(str(fold), 'models')
        if ~osp.exists(model_path):
            cmd = 'mkdir ' + model_path
            os.system(cmd)
            
        text_file = open(str(fold)+'/Result.txt', 'w')
        text_file.write('fold = ' + str(fold) + '\n')
        for nSample in sample_size:
            text_file.write('%g ' % nSample)
            
        batch_counter = 0
        save_index = 0
        test_result = np.array([]).reshape((0, 5, sample_size[1]))
        train_loss_list = []
        test_loss_list = []
        result_path = osp.join(str(fold), '/training_results')
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
                        
            """ training """
            for epoch in range(15):
                sess.run(train_iter.initializer)
                for _ in range(sample_size[0] // train_batch_size):
                    try:
                        batch_index = sess.run(train_element)
                    except tf.errors.OutOfRangeError:
                        break
                    
                    # train generator
                    batch_face = np.array(face_train[batch_index])
                    batch_left = np.array(left_train[batch_index])
                    batch_right = np.array(right_train[batch_index])
                    batch_vec = np.array(vec_train[batch_index])
                    batch_face, batch_eye = randomRotate(batch_face, batch_vec)
                    batch_subj = np.array(subj_train[batch_index])
                    batch_face,batch_left,batch_right,batch_eye,batch_subj_f = flip_images(batch_face,batch_left,
                                                                                           batch_right,batch_eye, batch_subj,
                                                                                           0.5,num_subj)
                    
                    # check training loss
                    if batch_counter % FLAGS.train_check_step == 0:
                        res_loss = sess.run(est_loss, feed_dict={
                                      x_f: batch_face, x_l: batch_left, x_r: batch_right, y_: batch_eye,
                                      keep_prob: 1.0,
                                      is_train: False,
                                      subj_id: dense_to_one_hot(batch_subj_f[:,1:2],n_classes=2*num_subj),
                                      is_aug: True})
    
                        print('Epoch %d %d, batch accuracy %g' % (epoch, batch_counter, np.mean(res_loss)))
                        text_file.write('Epoch %d %d, batch accuracy %g\n' % (epoch, batch_counter, np.mean(res_loss)))
                        train_loss_list.append(np.mean(res_loss))
                    
                    # update network
                    if batch_counter < FLAGS.warm_up:
                        sess.run(optimizer1, 
                                 feed_dict={x_f: batch_face, x_l: batch_left, x_r: batch_right,y_: batch_eye,
                                            keep_prob: 0.5,
                                            is_train: True,
                                            subj_id: dense_to_one_hot(batch_subj_f[:,1:2],n_classes=2*num_subj),
                                            is_aug: True})
                    else:
                        sess.run([optimizer1, optimizer2], 
                                 feed_dict={x_f: batch_face, x_l: batch_left, x_r: batch_right,y_: batch_eye,
                                            keep_prob: 0.5,
                                            is_train: True,
                                            subj_id: dense_to_one_hot(batch_subj_f[:,1:2],n_classes=2*num_subj),
                                            is_aug: True})

                    batch_counter += 1
                   
                    
                    """ testing accuracy """
                    if batch_counter % FLAGS.test_check_step == 0:
                        
                        test_result = np.concatenate([test_result, np.zeros((1, 5, sample_size[1]))], axis=0)
                        test_loss = np.array([], dtype = np.float32)
                        result_list1 = np.array([],dtype = np.float32).reshape((2,0))
                        result_gt = np.array([],dtype = np.float32).reshape((2,0))
                        
                        sess.run(test_iter.initializer)
                        for batch_step in range(eye_test.shape[0] // batch_size+1):
                            try:
                                batch_index = sess.run(test_element)
                            except tf.errors.OutOfRangeError:
                                break
                            batch_face = np.array(face_test[batch_index])
                            batch_left = np.array(left_test[batch_index])
                            batch_right = np.array(right_test[batch_index])
                            batch_eye = np.array(eye_test[batch_index])
        
                            res_loss, res_t_hat = sess.run([est_loss, t_hat], feed_dict={
                                                 x_f: batch_face, x_l: batch_left, x_r: batch_right,
                                                 y_: batch_eye,
                                                 keep_prob: 1.0,
                                                 is_train: False,
                                                 subj_id: batch_subj100,
                                                 is_aug: False})
                            res_t_hat /= 10.
                            test_loss = np.append(test_loss, np.mean(res_loss))
                            result_list1 = np.hstack([result_list1, res_t_hat.transpose()])
                            result_gt = np.hstack([result_gt, batch_eye.transpose()])
                           
                        test_result[save_index, 0:2, :] = result_gt
                        test_result[save_index, 2:4, :] = result_list1
                        angle_result = _angle2error(_2d2vec(result_list1.transpose()), vec_test)
                        test_result[save_index, 4, :] = angle_result
                        
                        text_file.write('testing accuracy %g %g\n' % (np.mean(test_loss), np.mean(angle_result)))
                        print('testing accuracy %g %g' % (np.mean(test_loss),np.mean(angle_result)))
                        test_loss_list.append(np.mean(test_loss))
                        save_index += 1

                    
                    if batch_counter % FLAGS.save_interval == 0:
                        print('save_index: %g' % save_index)
                        print(' ')
                        
                        #if np.mean(valid_loss) < prev_min:
                        #    save_path = saver.save(sess, str(fold)+"/models/model"+str(save_index)+".ckpt")
                        #prev_min = np.mean(valid_loss)
                        save_path = saver.save(sess, osp.join(model_path, 'model{}.ckpt'.format(save_index)))                                
                            
                        b_W_fc = bias_W_fc.eval()
                        np.savez(result_path, fold = fold,
                     test_result=test_result,
                     train_loss_list=train_loss_list,test_loss_list=test_loss_list,
                     b_W_fc=b_W_fc)
                    
                        
            save_path = saver.save(sess, osp.join(model_path, 'model{}.ckpt'.format(save_index)))    
            save_index += 1
            b_W_fc = bias_W_fc.eval()
            
            np.savez(result_path, fold = fold,
                     test_result=test_result,
                     train_loss_list=train_loss_list,test_loss_list=test_loss_list,
                     b_W_fc=b_W_fc)
                   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--MPII_dir', type=str,
                        default='/home/zchenbc/EyeContact/MPIIFaceGaze/data_normalization_face_eyeCenter/data_cv/',
                        help='Directory for storing Mpii data')
    parser.add_argument('--vgg_dir', type=str,
                        default='/home/zchenbc/EyeContact/vgg16_weights/vgg16_weights.npz',
                        help='Directory for pretrained vgg16')
    parser.add_argument('--train_check_step', type=int,
                        default=500,
                        help='The interval to print training loss')
    parser.add_argument('--test_check_step', type=int,
                        default=500,
                        help='The interval to print training loss')
    parser.add_argument('--warm_up', type=int,
                        default=500,
                        help='The step to freeze transfered layer')
    parser.add_argument('--save_interval', type=int,
                        default=500,
                        help='The interval to save model and results')
    

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
