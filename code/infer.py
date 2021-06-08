#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:31:47 2018

@author: zchenbc
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import dlib
import math
from GEDDnet import GEDDnet_infer
import PreProcess_eyecenter as PreP
import scipy.io as spio


def _2d2vec(input_angle):
    # change 2D angle to 3D vector
    # input_angle: (vertical, horizontal)
    vec = np.stack([np.sin(input_angle[:,1])*np.cos(input_angle[:,0]),
                    np.sin(input_angle[:,0]),
                    np.cos(input_angle[:,1])*np.cos(input_angle[:,0])],axis=1)
    return vec


def _angle2error(vec1, vec2):
    # calculate the angular difference between vec1 and vec2
    tmp = np.sum(vec1*vec2, axis = 1)
    tmp = np.maximum(tmp, -1.)
    cos_angle = np.minimum(tmp, 1.)
    angle = np.arccos(cos_angle) * 180. / np.pi
    return cos_angle, angle


def shape_to_np(shape):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=np.float32)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for ii in range(0, 68):
		coords[ii] = (shape.part(ii).x, shape.part(ii).y)

	# return the list of (x, y)-coordinates
	return coords


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(_):
    # get camera matrix
    dataset = spio.loadmat(FLAGS.camera_mat)
    cameraMat = dataset['camera_matrix']
    inv_cameraMat = np.linalg.inv(cameraMat)
    cam_new = np.mat([[1536., 0., 960.],[0., 1536., 540.],[0., 0., 1.]])
    cam_face = np.mat([[1536., 0., 48.],[0., 1536., 48.],[0., 0., 1.]])
    inv_cam_face = np.linalg.inv(cam_face)
    # define face detector and landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FLAGS.shape_predictor)
    cv2.namedWindow("win_frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("win_frame", 270, 180)
    cv2.namedWindow("win_face", cv2.WINDOW_NORMAL)
    cv2.moveWindow("win_face", 0, 500)
    cv2.resizeWindow("win_face", 300, 300)
    cv2.namedWindow("win_eye", cv2.WINDOW_NORMAL)
    cv2.moveWindow("win_eye", 500, 0)
    cv2.resizeWindow("win_eye", 2*192, 2*128)
    # define video capturer
    video_capture = cv2.VideoCapture(0)
    video_capture.set(39, 0)
    video_capture.set(3, 1920)
    video_capture.set(4, 1080)
    scale = 0.25
    input_size = (64, 96)
    gaze_lock = np.zeros(6, np.float64)
    gaze_unlock = np.zeros(15, np.float64)
    gaze_cursor = np.zeros(1, np.int_)
    shape = None
    face_backup = np.zeros((input_size[1], input_size[1], 3))
    left_backup = np.zeros((input_size[1], input_size[1], 3))
    rigt_backup = np.zeros((input_size[1], input_size[1], 3))
    # define model
    mu = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape(
        (1, 1, 3))

    # define network
    # input image
    x_f = tf.placeholder(tf.float32, [None, input_size[1], input_size[1], 3])
    x_l = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], 3])
    x_r = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], 3])

    y_conv, face_h_trans, h_trans = GEDDnet_infer(x_f, x_l, x_r, mu,
                                                  vgg_path=FLAGS.vgg_dir,
                                                  num_subj=FLAGS.num_subject)

    all_var = tf.global_variables()
    var_to_restore = []
    for var in all_var:
        if 'Momentum' in var.name or 'LearnRate' in var.name:
            continue
        var_to_restore.append(var)

    saver = tf.train.Saver(var_to_restore)

    TH = 0.99
    ref_vec = np.array([[0, 0, 1]])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model_path)
        counter = 0

        while True:
            # Capture frame-by-frame
            currFace = False
            ret, frame = video_capture.read()
            frame = frame[:, ::-1, :].copy()
            frame_small = cv2.resize(frame,
                                     None,
                                     fx=scale,
                                     fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray_big = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if counter % 2 == 0:
                rects = detector(gray, 1)
            # loop over the face detections

            for (ii, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                if ii == 0:
                    tmp = np.array([rect.left(), rect.top(), rect.right(),rect.bottom()]) / scale
                    tmp = tmp.astype(np.long)
                    rect_new = dlib.rectangle(tmp[0], tmp[1], tmp[2], tmp[3])
                    shape = predictor(gray_big, rect_new)
                    shape = shape_to_np(shape)
                    shape = shape.astype(np.int_)
                    currFace = True
                    #for (x, y) in shape:
                        #cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            # cut the face
            if shape is None:
                pass
            else:
                face_img, left_img, rigt_img, eye_lm, fc_c_world, warpMat = PreP.WarpNCrop(frame[:,:,::-1], shape, inv_cameraMat, cam_new)

                if face_img.shape[0:2] == (input_size[1], input_size[1]) and \
                   left_img.shape[0:2] == input_size and \
                   rigt_img.shape[0:2] == input_size:

                    face_backup = face_img
                    left_backup = left_img
                    rigt_backup = rigt_img
                else:
                    face_img = face_backup
                    left_img = left_backup
                    rigt_img = rigt_backup

                y_result, eye_img, face_img = sess.run([y_conv, h_trans, face_h_trans], feed_dict={
                                                       x_f: face_img[None, :],
                                                       x_l: left_img[None, :],
                                                       x_r: rigt_img[None, :]})
                y_vec = _2d2vec(y_result)
                y_binary, y_angle = _angle2error(y_vec, ref_vec)
                gaze_lock_avg = y_binary[0]
                gaze_cursor += 1

                #####################
                # insert your control code here, y_result is the gaze (pitch, yaw) in radis
                #ser = serial.Serial('/dev/ttyUSB0', 9600)

                yaw = y_result[0][0] * (180 / math.pi)
                pitch = y_result[0][1] * (180 / math.pi)

                print('pitch', pitch)
                print('yaw', yaw)
                ### If you want to calculate the gaze vector in the original images
                ### you should use the warpMat

                ####################
                face_img_gaze = PreP.WarpNDraw(face_img[0], eye_lm, y_vec, cam_face, inv_cam_face)
                cv2.circle(frame_small, (int(960*scale), int(540*scale)), 3, (255, 0, 0), -1)
                #w, h,_=frame_small.shape
                #cv2.putText(frame_small, '{}-{}'.format(y_vec[0][0], y_vec[0][1]),(15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255, 255))
                cv2.imshow('win_frame', frame_small)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                #r = np.max([gaze_lock_avg, 0.01])
                if gaze_lock_avg >= TH:
                    border_color = np.array([255,255,255]) - (gaze_lock_avg - TH)/(1-TH) * np.array([0, 255, 255])

                else:
                    border_color = np.array([255,255,255]) - (TH-gaze_lock_avg )/TH * np.array([255, 255, 0])

                face_img = cv2.copyMakeBorder(face_img_gaze, 5, 5, 5, 5,
                                                    cv2.BORDER_CONSTANT, value=border_color)

                cv2.imshow('win_face', face_img / 255.)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cv2.imshow('win_eye', eye_img[0] / 255.)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                counter += 1

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vgg_dir', type=str,
                        default='../data/vgg16_weights.npz',
                        help='Directory for pretrained vgg16')

    parser.add_argument('--model_path', type=str,
                        default='../data/models/model19.ckpt',
                        help='Path of the trained model')

    parser.add_argument('--num_subject', type=int,
                        default=50,
                        help='The total number of subject (not include horizontal flip)')

    parser.add_argument("--shape-predictor", type=str,
                        default='shape_predictor_68_face_landmarks.dat',
	                    help="Path to facial landmark predictor")

    parser.add_argument("--camera_mat", type=str,
                        default='../data/camera_matrix.mat',
	                    help="Path to camera matrix")
    FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
