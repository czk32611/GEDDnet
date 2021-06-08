#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:11:37 2020

@author: zchenbc
"""

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

import sys

import tensorflow as tf
import numpy as np
from tf_utils import conv2d, dilated2d, max_pool_2x2, weight_variable, bias_variable


FLAGS = None


def GEDDnet(face,
            left_eye,
            right_eye,
            keep_prob,
            is_train,
            subj_id,
            vgg_path,
            num_subj,
            rf=[[2, 2], [3, 3], [5, 5], [11, 11]],
            num_face=[64, 128, 64, 64, 128, 256, 64],
            r=[[2, 2], [3, 3], [4, 5], [5, 11]],
            num_eye=[64, 128, 64, 64, 128, 256],
            num_comb=[0, 256]):

    num_comb[0] = num_face[-1]+2*num_eye[-1]

    vgg = np.load(vgg_path)
    with tf.variable_scope("transfer"):
        W_conv1_1 = tf.Variable(vgg['conv1_1_W'])
        b_conv1_1 = tf.Variable(vgg['conv1_1_b'])
        W_conv1_2 = tf.Variable(vgg['conv1_2_W'])
        b_conv1_2 = tf.Variable(vgg['conv1_2_b'])

        W_conv2_1 = tf.Variable(vgg['conv2_1_W'])
        b_conv2_1 = tf.Variable(vgg['conv2_1_b'])
        W_conv2_2 = tf.Variable(vgg['conv2_2_W'])
        b_conv2_2 = tf.Variable(vgg['conv2_2_b'])
    del vgg

    """ define network """
    # face
    face_h_conv1_1 = tf.nn.relu(conv2d(face, W_conv1_1) + b_conv1_1)
    face_h_conv1_2 = tf.nn.relu(conv2d(face_h_conv1_1, W_conv1_2) + b_conv1_2)
    face_h_pool1 = max_pool_2x2(face_h_conv1_2)

    face_h_conv2_1 = tf.nn.relu(conv2d(face_h_pool1, W_conv2_1) + b_conv2_1)
    face_h_conv2_2 = tf.nn.relu(conv2d(face_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.


    with tf.variable_scope("face"):

        face_W_conv2_3 = weight_variable([1, 1, num_face[1], num_face[2]], std=0.125)
        face_b_conv2_3 = bias_variable([num_face[2]], std=0.001)

        face_W_conv3_1 = weight_variable([3, 3, num_face[2], num_face[3]], std=0.06)
        face_b_conv3_1 = bias_variable([num_face[3]], std=0.001)
        face_W_conv3_2 = weight_variable([3, 3, num_face[3], num_face[3]], std=0.06)
        face_b_conv3_2 = bias_variable([num_face[3]], std=0.001)

        face_W_conv4_1 = weight_variable([3, 3, num_face[3], num_face[4]], std=0.08)
        face_b_conv4_1 = bias_variable([num_face[4]], std=0.001)
        face_W_conv4_2 = weight_variable([3, 3, num_face[4], num_face[4]], std=0.07)
        face_b_conv4_2 = bias_variable([num_face[4]], std=0.001)

        face_W_fc1 = weight_variable([6*6*num_face[4], num_face[5]], std=0.035)
        face_b_fc1 = bias_variable([num_face[5]], std=0.001)

        face_W_fc2 = weight_variable([num_face[5], num_face[6]], std=0.1)
        face_b_fc2 = bias_variable([num_face[6]], std=0.001)

        face_h_conv2_3 = tf.nn.relu(conv2d(face_h_conv2_2, face_W_conv2_3) + face_b_conv2_3)
        face_h_conv2_3_norm = tf.layers.batch_normalization(face_h_conv2_3, training=is_train, scale=False, renorm=True,
                                                            name="f_conv2_3")

        face_h_conv3_1 = tf.nn.relu(dilated2d(face_h_conv2_3_norm, face_W_conv3_1, rf[0]) + face_b_conv3_1)
        face_h_conv3_1_norm = tf.layers.batch_normalization(face_h_conv3_1, training=is_train, scale=False, renorm=True,
                                                            name="f_conv3_1")

        face_h_conv3_2 = tf.nn.relu(dilated2d(face_h_conv3_1_norm, face_W_conv3_2, rf[1]) + face_b_conv3_2)
        face_h_conv3_2_norm = tf.layers.batch_normalization(face_h_conv3_2, training=is_train, scale=False, renorm=True,
                                                            name="f_conv3_2")

        face_h_conv4_1 = tf.nn.relu(dilated2d(face_h_conv3_2_norm, face_W_conv4_1, rf[2]) + face_b_conv4_1)
        face_h_conv4_1_norm = tf.layers.batch_normalization(face_h_conv4_1, training=is_train, scale=False, renorm=True,
                                                            name="f_conv4_1")

        face_h_conv4_2 = tf.nn.relu(dilated2d(face_h_conv4_1_norm, face_W_conv4_2, rf[3]) + face_b_conv4_2)
        face_h_conv4_2_norm = tf.layers.batch_normalization(face_h_conv4_2, training=is_train, scale=False, renorm=True,
                                                            name="f_conv4_2")

        face_h_pool4_flat = tf.reshape(face_h_conv4_2_norm, [-1, 6*6*num_face[4]])

        face_h_fc1 = tf.nn.relu(tf.matmul(face_h_pool4_flat, face_W_fc1) + face_b_fc1)
        face_h_fc1_norm = tf.layers.batch_normalization(face_h_fc1, training=is_train, scale=False, renorm=True,
                                                        name="f_fc1")
        face_h_fc1_drop = tf.nn.dropout(face_h_fc1_norm, keep_prob)

        face_h_fc2 = tf.nn.relu(tf.matmul(face_h_fc1_drop, face_W_fc2) + face_b_fc2)
        face_h_fc2_norm = tf.layers.batch_normalization(face_h_fc2, training=is_train, scale=False, renorm=True,
                                                        name="f_fc2")


    eye1_h_conv1_1 = tf.nn.relu(conv2d(left_eye, W_conv1_1) + b_conv1_1)
    eye1_h_conv1_2 = tf.nn.relu(conv2d(eye1_h_conv1_1, W_conv1_2) + b_conv1_2)
    eye1_h_pool1 = max_pool_2x2(eye1_h_conv1_2)

    eye1_h_conv2_1 = tf.nn.relu(conv2d(eye1_h_pool1, W_conv2_1) + b_conv2_1)
    eye1_h_conv2_2 = tf.nn.relu(conv2d(eye1_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.

    eye2_h_conv1_1 = tf.nn.relu(conv2d(right_eye, W_conv1_1) + b_conv1_1)
    eye2_h_conv1_2 = tf.nn.relu(conv2d(eye2_h_conv1_1, W_conv1_2) + b_conv1_2)
    eye2_h_pool1 = max_pool_2x2(eye2_h_conv1_2)

    eye2_h_conv2_1 = tf.nn.relu(conv2d(eye2_h_pool1, W_conv2_1) + b_conv2_1)
    eye2_h_conv2_2 = tf.nn.relu(conv2d(eye2_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.


    with tf.variable_scope("eye"):
        # left eye
        eye_W_conv2_3 = weight_variable([1, 1, num_eye[1], num_eye[2]], std=0.125)
        eye_b_conv2_3 = bias_variable([num_eye[2]], std=0.001)

        eye_W_conv3_1 = weight_variable([3, 3, num_eye[2], num_eye[3]], std=0.06)
        eye_b_conv3_1 = bias_variable([num_eye[3]], std=0.001)
        eye_W_conv3_2 = weight_variable([3, 3, num_eye[3], num_eye[3]], std=0.06)
        eye_b_conv3_2 = bias_variable([num_eye[3]], std=0.001)

        eye_W_conv4_1 = weight_variable([3, 3, num_eye[3], num_eye[4]], std=0.06)
        eye_b_conv4_1 = bias_variable([num_eye[4]], std=0.001)
        eye_W_conv4_2 = weight_variable([3, 3, num_eye[4], num_eye[4]], std=0.04)
        eye_b_conv4_2 = bias_variable([num_eye[4]], std=0.001)

        eye1_W_fc1 = weight_variable([4*6*num_eye[4], num_eye[5]], std=0.026)
        eye1_b_fc1 = bias_variable([num_eye[5]], std=0.001)

        eye2_W_fc1 = weight_variable([4*6*num_eye[4], num_eye[5]], std=0.026)
        eye2_b_fc1 = bias_variable([num_eye[5]], std=0.001)

        eye1_h_conv2_3 = tf.nn.relu(conv2d(eye1_h_conv2_2, eye_W_conv2_3) + eye_b_conv2_3)
        eye1_h_conv2_3_norm = tf.layers.batch_normalization(eye1_h_conv2_3, training=is_train, scale=False, renorm=True,
                                                            name="e_conv2_3")

        eye1_h_conv3_1 = tf.nn.relu(dilated2d(eye1_h_conv2_3_norm, eye_W_conv3_1, r[0]) + eye_b_conv3_1)
        eye1_h_conv3_1_norm = tf.layers.batch_normalization(eye1_h_conv3_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_1")

        eye1_h_conv3_2 = tf.nn.relu(dilated2d(eye1_h_conv3_1_norm, eye_W_conv3_2, r[1]) + eye_b_conv3_2)
        eye1_h_conv3_2_norm = tf.layers.batch_normalization(eye1_h_conv3_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_2")

        eye1_h_conv4_1 = tf.nn.relu(dilated2d(eye1_h_conv3_2_norm, eye_W_conv4_1, r[2]) + eye_b_conv4_1)
        eye1_h_conv4_1_norm = tf.layers.batch_normalization(eye1_h_conv4_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_1")

        eye1_h_conv4_2 = tf.nn.relu(dilated2d(eye1_h_conv4_1_norm, eye_W_conv4_2, r[3]) + eye_b_conv4_2)
        eye1_h_conv4_2_norm = tf.layers.batch_normalization(eye1_h_conv4_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_2")

        eye1_h_pool4_flat = tf.reshape(eye1_h_conv4_2_norm, [-1, 4*6*num_eye[4]])

        eye1_h_fc1 = tf.nn.relu(tf.matmul(eye1_h_pool4_flat, eye1_W_fc1) + eye1_b_fc1)
        eye1_h_fc1_norm = tf.layers.batch_normalization(eye1_h_fc1, training=is_train, scale=False, renorm=True,
                                                        name="e1_fc1")

        # right eye
        eye2_h_conv2_3 = tf.nn.relu(conv2d(eye2_h_conv2_2, eye_W_conv2_3) + eye_b_conv2_3)
        eye2_h_conv2_3_norm = tf.layers.batch_normalization(eye2_h_conv2_3, training=is_train, scale=False, renorm=True,
                                                            name="e_conv2_3", reuse=True)

        eye2_h_conv3_1 = tf.nn.relu(dilated2d(eye2_h_conv2_3_norm, eye_W_conv3_1, r[0]) + eye_b_conv3_1)
        eye2_h_conv3_1_norm = tf.layers.batch_normalization(eye2_h_conv3_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_1", reuse=True)

        eye2_h_conv3_2 = tf.nn.relu(dilated2d(eye2_h_conv3_1_norm, eye_W_conv3_2, r[1]) + eye_b_conv3_2)
        eye2_h_conv3_2_norm = tf.layers.batch_normalization(eye2_h_conv3_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_2", reuse=True)

        eye2_h_conv4_1 = tf.nn.relu(dilated2d(eye2_h_conv3_2_norm, eye_W_conv4_1, r[2]) + eye_b_conv4_1)
        eye2_h_conv4_1_norm = tf.layers.batch_normalization(eye2_h_conv4_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_1", reuse=True)

        eye2_h_conv4_2 = tf.nn.relu(dilated2d(eye2_h_conv4_1_norm, eye_W_conv4_2, r[3]) + eye_b_conv4_2)
        eye2_h_conv4_2_norm = tf.layers.batch_normalization(eye2_h_conv4_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_2", reuse=True)

        eye2_h_pool4_flat = tf.reshape(eye2_h_conv4_2_norm, [-1, 4*6*num_eye[4]])

        eye2_h_fc1 = tf.nn.relu(tf.matmul(eye2_h_pool4_flat, eye2_W_fc1) + eye2_b_fc1)
        eye2_h_fc1_norm = tf.layers.batch_normalization(eye2_h_fc1, training=is_train, scale=False, renorm=True,
                                                        name="e2_fc1")

    # combine both eyes and face
    with tf.variable_scope("combine"):

        cls1_W_fc2 = weight_variable([num_comb[0], num_comb[1]], std=0.07)
        cls1_b_fc2 = bias_variable([num_comb[1]], std=0.001)

        cls1_W_fc3 = weight_variable([num_comb[1], 2], std=0.125)
        cls1_b_fc3 = bias_variable([2], std=0.001)

        cls1_h_fc1_norm = tf.concat([face_h_fc2_norm, eye1_h_fc1_norm, eye2_h_fc1_norm], axis=1)
        cls1_h_fc1_drop = tf.nn.dropout(cls1_h_fc1_norm, keep_prob)
        cls1_h_fc2 = tf.nn.relu(tf.matmul(cls1_h_fc1_drop, cls1_W_fc2) + cls1_b_fc2)
        cls1_h_fc2_norm = tf.layers.batch_normalization(cls1_h_fc2, training=is_train, scale=False, renorm=True,
                                                        name="c_fc2")
        cls1_h_fc2_drop = tf.nn.dropout(cls1_h_fc2_norm, keep_prob)

        t_hat = tf.matmul(cls1_h_fc2_drop, cls1_W_fc3) + cls1_b_fc3


    """ bias learning from subject id """
    num_bias = (2 * num_subj,)
    with tf.variable_scope("bias"):

        bias_W_fc = weight_variable([num_bias[0], 2], std=0.125)
        b_hat = tf.matmul(subj_id, bias_W_fc)

    g_hat = t_hat + b_hat

    l2_loss = (1e-2*tf.nn.l2_loss(W_conv1_1) +
               1e-2*tf.nn.l2_loss(W_conv1_2) +
               1e-2*tf.nn.l2_loss(W_conv2_1) +
               1e-2*tf.nn.l2_loss(W_conv2_2) +
               tf.nn.l2_loss(face_W_conv2_3) +
               tf.nn.l2_loss(face_W_conv3_1) +
               tf.nn.l2_loss(face_W_conv3_2) +
               tf.nn.l2_loss(face_W_conv4_1) +
               tf.nn.l2_loss(face_W_conv4_2) +
               tf.nn.l2_loss(face_W_fc1) +
               tf.nn.l2_loss(face_W_fc2) +
               tf.nn.l2_loss(eye_W_conv2_3) +
               tf.nn.l2_loss(eye_W_conv3_1) +
               tf.nn.l2_loss(eye_W_conv3_2) +
               tf.nn.l2_loss(eye_W_conv4_1) +
               tf.nn.l2_loss(eye_W_conv4_2) +
               tf.nn.l2_loss(eye1_W_fc1) +
               tf.nn.l2_loss(eye2_W_fc1) +
               tf.nn.l2_loss(cls1_W_fc2) +
               tf.nn.l2_loss(cls1_W_fc3))

    return g_hat, t_hat, bias_W_fc, l2_loss


def GEDDnet_infer(face,
                  left_eye,
                  right_eye,
                  mu,
                  vgg_path,
                  num_subj,
                  keep_prob=1.0,
                  is_train=False,
                  rf=[[2, 2], [3, 3], [5, 5], [11, 11]],
                  num_face=[64, 128, 64, 64, 128, 256, 64],
                  r=[[2, 2], [3, 3], [4, 5], [5, 11]],
                  num_eye=[64, 128, 64, 64, 128, 256],
                  num_comb=[0, 256]):

    num_comb[0] = num_face[-1]+2*num_eye[-1]

    vgg = np.load(vgg_path)
    with tf.variable_scope("transfer"):
        W_conv1_1 = tf.Variable(vgg['conv1_1_W'])
        b_conv1_1 = tf.Variable(vgg['conv1_1_b'])
        W_conv1_2 = tf.Variable(vgg['conv1_2_W'])
        b_conv1_2 = tf.Variable(vgg['conv1_2_b'])

        W_conv2_1 = tf.Variable(vgg['conv2_1_W'])
        b_conv2_1 = tf.Variable(vgg['conv2_1_b'])
        W_conv2_2 = tf.Variable(vgg['conv2_2_W'])
        b_conv2_2 = tf.Variable(vgg['conv2_2_b'])
    del vgg

    """ define network """
    # face
    face_h_conv1_1 = tf.nn.relu(conv2d(face - mu, W_conv1_1) + b_conv1_1)
    face_h_conv1_2 = tf.nn.relu(conv2d(face_h_conv1_1, W_conv1_2) + b_conv1_2)
    face_h_pool1 = max_pool_2x2(face_h_conv1_2)

    face_h_conv2_1 = tf.nn.relu(conv2d(face_h_pool1, W_conv2_1) + b_conv2_1)
    face_h_conv2_2 = tf.nn.relu(conv2d(face_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.


    with tf.variable_scope("face"):

        face_W_conv2_3 = weight_variable([1, 1, num_face[1], num_face[2]], std=0.125)
        face_b_conv2_3 = bias_variable([num_face[2]], std=0.001)

        face_W_conv3_1 = weight_variable([3, 3, num_face[2], num_face[3]], std=0.06)
        face_b_conv3_1 = bias_variable([num_face[3]], std=0.001)
        face_W_conv3_2 = weight_variable([3, 3, num_face[3], num_face[3]], std=0.06)
        face_b_conv3_2 = bias_variable([num_face[3]], std=0.001)

        face_W_conv4_1 = weight_variable([3, 3, num_face[3], num_face[4]], std=0.08)
        face_b_conv4_1 = bias_variable([num_face[4]], std=0.001)
        face_W_conv4_2 = weight_variable([3, 3, num_face[4], num_face[4]], std=0.07)
        face_b_conv4_2 = bias_variable([num_face[4]], std=0.001)

        face_W_fc1 = weight_variable([6*6*num_face[4], num_face[5]], std=0.035)
        face_b_fc1 = bias_variable([num_face[5]], std=0.001)

        face_W_fc2 = weight_variable([num_face[5], num_face[6]], std=0.1)
        face_b_fc2 = bias_variable([num_face[6]], std=0.001)

        face_h_conv2_3 = tf.nn.relu(conv2d(face_h_conv2_2, face_W_conv2_3) + face_b_conv2_3)
        face_h_conv2_3_norm = tf.layers.batch_normalization(face_h_conv2_3, training=is_train, scale=False, renorm=True,
                                                            name="f_conv2_3")

        face_h_conv3_1 = tf.nn.relu(dilated2d(face_h_conv2_3_norm, face_W_conv3_1, rf[0]) + face_b_conv3_1)
        face_h_conv3_1_norm = tf.layers.batch_normalization(face_h_conv3_1, training=is_train, scale=False, renorm=True,
                                                            name="f_conv3_1")

        face_h_conv3_2 = tf.nn.relu(dilated2d(face_h_conv3_1_norm, face_W_conv3_2, rf[1]) + face_b_conv3_2)
        face_h_conv3_2_norm = tf.layers.batch_normalization(face_h_conv3_2, training=is_train, scale=False, renorm=True,
                                                            name="f_conv3_2")

        face_h_conv4_1 = tf.nn.relu(dilated2d(face_h_conv3_2_norm, face_W_conv4_1, rf[2]) + face_b_conv4_1)
        face_h_conv4_1_norm = tf.layers.batch_normalization(face_h_conv4_1, training=is_train, scale=False, renorm=True,
                                                            name="f_conv4_1")

        face_h_conv4_2 = tf.nn.relu(dilated2d(face_h_conv4_1_norm, face_W_conv4_2, rf[3]) + face_b_conv4_2)
        face_h_conv4_2_norm = tf.layers.batch_normalization(face_h_conv4_2, training=is_train, scale=False, renorm=True,
                                                            name="f_conv4_2")

        face_h_pool4_flat = tf.reshape(face_h_conv4_2_norm, [-1, 6*6*num_face[4]])

        face_h_fc1 = tf.nn.relu(tf.matmul(face_h_pool4_flat, face_W_fc1) + face_b_fc1)
        face_h_fc1_norm = tf.layers.batch_normalization(face_h_fc1, training=is_train, scale=False, renorm=True,
                                                        name="f_fc1")
        face_h_fc1_drop = tf.nn.dropout(face_h_fc1_norm, keep_prob)

        face_h_fc2 = tf.nn.relu(tf.matmul(face_h_fc1_drop, face_W_fc2) + face_b_fc2)
        face_h_fc2_norm = tf.layers.batch_normalization(face_h_fc2, training=is_train, scale=False, renorm=True,
                                                        name="f_fc2")


    eye1_h_conv1_1 = tf.nn.relu(conv2d(left_eye - mu, W_conv1_1) + b_conv1_1)
    eye1_h_conv1_2 = tf.nn.relu(conv2d(eye1_h_conv1_1, W_conv1_2) + b_conv1_2)
    eye1_h_pool1 = max_pool_2x2(eye1_h_conv1_2)

    eye1_h_conv2_1 = tf.nn.relu(conv2d(eye1_h_pool1, W_conv2_1) + b_conv2_1)
    eye1_h_conv2_2 = tf.nn.relu(conv2d(eye1_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.

    eye2_h_conv1_1 = tf.nn.relu(conv2d(right_eye - mu, W_conv1_1) + b_conv1_1)
    eye2_h_conv1_2 = tf.nn.relu(conv2d(eye2_h_conv1_1, W_conv1_2) + b_conv1_2)
    eye2_h_pool1 = max_pool_2x2(eye2_h_conv1_2)

    eye2_h_conv2_1 = tf.nn.relu(conv2d(eye2_h_pool1, W_conv2_1) + b_conv2_1)
    eye2_h_conv2_2 = tf.nn.relu(conv2d(eye2_h_conv2_1, W_conv2_2) + b_conv2_2) / 100.


    with tf.variable_scope("eye"):
        # left eye
        eye_W_conv2_3 = weight_variable([1, 1, num_eye[1], num_eye[2]], std=0.125)
        eye_b_conv2_3 = bias_variable([num_eye[2]], std=0.001)

        eye_W_conv3_1 = weight_variable([3, 3, num_eye[2], num_eye[3]], std=0.06)
        eye_b_conv3_1 = bias_variable([num_eye[3]], std=0.001)
        eye_W_conv3_2 = weight_variable([3, 3, num_eye[3], num_eye[3]], std=0.06)
        eye_b_conv3_2 = bias_variable([num_eye[3]], std=0.001)

        eye_W_conv4_1 = weight_variable([3, 3, num_eye[3], num_eye[4]], std=0.06)
        eye_b_conv4_1 = bias_variable([num_eye[4]], std=0.001)
        eye_W_conv4_2 = weight_variable([3, 3, num_eye[4], num_eye[4]], std=0.04)
        eye_b_conv4_2 = bias_variable([num_eye[4]], std=0.001)

        eye1_W_fc1 = weight_variable([4*6*num_eye[4], num_eye[5]], std=0.026)
        eye1_b_fc1 = bias_variable([num_eye[5]], std=0.001)

        eye2_W_fc1 = weight_variable([4*6*num_eye[4], num_eye[5]], std=0.026)
        eye2_b_fc1 = bias_variable([num_eye[5]], std=0.001)

        eye1_h_conv2_3 = tf.nn.relu(conv2d(eye1_h_conv2_2, eye_W_conv2_3) + eye_b_conv2_3)
        eye1_h_conv2_3_norm = tf.layers.batch_normalization(eye1_h_conv2_3, training=is_train, scale=False, renorm=True,
                                                            name="e_conv2_3")

        eye1_h_conv3_1 = tf.nn.relu(dilated2d(eye1_h_conv2_3_norm, eye_W_conv3_1, r[0]) + eye_b_conv3_1)
        eye1_h_conv3_1_norm = tf.layers.batch_normalization(eye1_h_conv3_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_1")

        eye1_h_conv3_2 = tf.nn.relu(dilated2d(eye1_h_conv3_1_norm, eye_W_conv3_2, r[1]) + eye_b_conv3_2)
        eye1_h_conv3_2_norm = tf.layers.batch_normalization(eye1_h_conv3_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_2")

        eye1_h_conv4_1 = tf.nn.relu(dilated2d(eye1_h_conv3_2_norm, eye_W_conv4_1, r[2]) + eye_b_conv4_1)
        eye1_h_conv4_1_norm = tf.layers.batch_normalization(eye1_h_conv4_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_1")

        eye1_h_conv4_2 = tf.nn.relu(dilated2d(eye1_h_conv4_1_norm, eye_W_conv4_2, r[3]) + eye_b_conv4_2)
        eye1_h_conv4_2_norm = tf.layers.batch_normalization(eye1_h_conv4_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_2")

        eye1_h_pool4_flat = tf.reshape(eye1_h_conv4_2_norm, [-1, 4*6*num_eye[4]])

        eye1_h_fc1 = tf.nn.relu(tf.matmul(eye1_h_pool4_flat, eye1_W_fc1) + eye1_b_fc1)
        eye1_h_fc1_norm = tf.layers.batch_normalization(eye1_h_fc1, training=is_train, scale=False, renorm=True,
                                                        name="e1_fc1")

        # right eye
        eye2_h_conv2_3 = tf.nn.relu(conv2d(eye2_h_conv2_2, eye_W_conv2_3) + eye_b_conv2_3)
        eye2_h_conv2_3_norm = tf.layers.batch_normalization(eye2_h_conv2_3, training=is_train, scale=False, renorm=True,
                                                            name="e_conv2_3", reuse=True)

        eye2_h_conv3_1 = tf.nn.relu(dilated2d(eye2_h_conv2_3_norm, eye_W_conv3_1, r[0]) + eye_b_conv3_1)
        eye2_h_conv3_1_norm = tf.layers.batch_normalization(eye2_h_conv3_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_1", reuse=True)

        eye2_h_conv3_2 = tf.nn.relu(dilated2d(eye2_h_conv3_1_norm, eye_W_conv3_2, r[1]) + eye_b_conv3_2)
        eye2_h_conv3_2_norm = tf.layers.batch_normalization(eye2_h_conv3_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv3_2", reuse=True)

        eye2_h_conv4_1 = tf.nn.relu(dilated2d(eye2_h_conv3_2_norm, eye_W_conv4_1, r[2]) + eye_b_conv4_1)
        eye2_h_conv4_1_norm = tf.layers.batch_normalization(eye2_h_conv4_1, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_1", reuse=True)

        eye2_h_conv4_2 = tf.nn.relu(dilated2d(eye2_h_conv4_1_norm, eye_W_conv4_2, r[3]) + eye_b_conv4_2)
        eye2_h_conv4_2_norm = tf.layers.batch_normalization(eye2_h_conv4_2, training=is_train, scale=False, renorm=True,
                                                            name="e_conv4_2", reuse=True)

        eye2_h_pool4_flat = tf.reshape(eye2_h_conv4_2_norm, [-1, 4*6*num_eye[4]])

        eye2_h_fc1 = tf.nn.relu(tf.matmul(eye2_h_pool4_flat, eye2_W_fc1) + eye2_b_fc1)
        eye2_h_fc1_norm = tf.layers.batch_normalization(eye2_h_fc1, training=is_train, scale=False, renorm=True,
                                                        name="e2_fc1")

    # combine both eyes and face
    with tf.variable_scope("combine"):

        cls1_W_fc2 = weight_variable([num_comb[0], num_comb[1]], std=0.07)
        cls1_b_fc2 = bias_variable([num_comb[1]], std=0.001)

        cls1_W_fc3 = weight_variable([num_comb[1], 2], std=0.125)
        cls1_b_fc3 = bias_variable([2], std=0.001)

        cls1_h_fc1_norm = tf.concat([face_h_fc2_norm, eye1_h_fc1_norm, eye2_h_fc1_norm], axis = 1)
        cls1_h_fc1_drop = tf.nn.dropout(cls1_h_fc1_norm, keep_prob)
        cls1_h_fc2 = tf.nn.relu(tf.matmul(cls1_h_fc1_drop, cls1_W_fc2) + cls1_b_fc2)
        cls1_h_fc2_norm = tf.layers.batch_normalization(cls1_h_fc2, training=is_train, scale=False, renorm=True,
                                                        name="c_fc2")
        cls1_h_fc2_drop = tf.nn.dropout(cls1_h_fc2_norm, keep_prob)

        t_hat = tf.matmul(cls1_h_fc2_drop, cls1_W_fc3) + cls1_b_fc3


    """ bias learning from subject id """
    num_bias = (2*num_subj,)
    with tf.variable_scope("bias"):
        bias_W_fc = weight_variable([num_bias[0], 2], std=0.125)

    g_hat = t_hat / 10.

    num_batch = 1
    face_h_trans = face + mu
    face2 = tf.image.resize_images(face, [64, 64])
    h_trans = tf.concat([face2+mu, tf.ones([num_batch, 64, 1, 3]),
                         left_eye+mu, tf.ones([num_batch, 64, 1, 3]),
                         right_eye+mu], axis = 2)


    return g_hat, face_h_trans, h_trans
