# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

# %% Borrowed utils from here: https://github.com/pkmital/tensorflow_tutorials/
import tensorflow as tf
import numpy as np
from math import sqrt

def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=strides,padding=padding)

def dilated2d(x, W, rate, strides=[1, 1, 1, 1], padding='VALID'):
    """conv2d returns a 2d convolution layer with full stride. depend on rate"""
    return tf.nn.convolution(x, W, strides=strides, padding=padding,
                             dilations=[1, rate[0], rate[1], 1])

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
# %%
def weight_variable(shape, std=0.1, trainable=True):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial, trainable=trainable)

# %%
def bias_variable(shape, std=0.1, trainable=True):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=std)
    return tf.Variable(tf.abs(initial), trainable=trainable)

# %%
def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    index = index_offset + labels.ravel()
    index = index.astype(np.int_)
    labels_one_hot.flat[index] = 1
    return labels_one_hot
