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


def grayNhist(input_image):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    gray_img = gray_img.astype(np.uint8)
    equ = cv2.equalizeHist(gray_img)
    output_image = np.stack((equ,equ,equ), axis = 2)
    return output_image


def point_to_matrix(points, desiredDist = 224 * 0.6):
    # obtain the affine matrix given two points
    points = points.astype(np.int)
    dX = points[2] - points[0]
    dY = points[3] - points[1]
    angle = np.degrees(np.arctan2(dY, dX))
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desiredDist / dist
    eyesCenter = ((points[0] + points[2]) // 2,
                  (points[1] + points[3]) // 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D((79.5, 79.5), angle, scale)
    return M, scale



def WarpNCrop(input_image, landmarks, inv_CameraMat, cam_new):
    """ crop images by perspective warping """
    eye_lm = np.vstack([landmarks[36:48, :].transpose(), np.ones((1,12))]) # eye landmarks
    eye_lm_w = np.dot(inv_CameraMat, eye_lm) # in world coordinate

    rigtEye_w = eye_lm_w[:,0:6]
    leftEye_w = eye_lm_w[:,6:]
    rigtEye_wc = np.mean(rigtEye_w, axis=1, keepdims=True)
    leftEye_wc = np.mean(leftEye_w, axis=1, keepdims=True)
    # warp the images to make the face at the center

    fc_c_world = np.mean(eye_lm_w, axis=1, keepdims=True)
    # calculate the rotation matrix
    forward = fc_c_world / np.linalg.norm(fc_c_world)
    hRx = leftEye_wc - rigtEye_wc
    down = np.cross(forward.reshape(-1), hRx.reshape(-1))
    down = down / np.linalg.norm(down)
    down.shape = (3,-1)
    right = np.cross(down.reshape(-1), forward.reshape(-1))
    right = right / np.linalg.norm(right)
    right.shape = (3,-1)

    rotMat = np.hstack([right, down, forward])
    #rotMat = np.eye(3)
    warpMat = cam_new * np.mat(rotMat.transpose()) * inv_CameraMat
    warped_image = cv2.warpPerspective(input_image, warpMat, input_image.shape[1::-1])

    # obtain the warped landmarks
    eye_lm_warped = warpMat * eye_lm
    eye_lm_warped = np.array(eye_lm_warped[:2] / eye_lm_warped[2]).astype(np.int_)
    eye_lm_store = np.mean(eye_lm_warped, axis=1)[1]

    #img_size = warped_image.shape
    """ plot landmarks
    cv2.circle(warped_image, (img_size[1]//2, img_size[0]//2), 10, (0, 0, 255), -1)
    for ii in range(eye_lm_warped.shape[1]):
        cv2.circle(warped_image, (eye_lm_warped[0,ii], eye_lm_warped[1,ii]), 10, (255, 0, 0), -1)
    """
    rigtEye_warped = eye_lm_warped[:,0:6]
    leftEye_warped = eye_lm_warped[:,6:]
    rigtEye_warpedc = np.mean(rigtEye_warped, axis=1)
    leftEye_warpedc = np.mean(leftEye_warped, axis=1)
    # obtain face image
    scale = 32. / np.abs(leftEye_warpedc[0]-rigtEye_warpedc[0])
    face_image = cv2.resize(warped_image, (0, 0),
                           fx=scale, fy=scale,interpolation = cv2.INTER_CUBIC)
    f_size = face_image.shape
    face_image = face_image[(f_size[0]//2-60+5):(f_size[0]//2+60+5), (f_size[1]//2-60):(f_size[1]//2+60)]
    face_image = grayNhist(face_image)
    face_image = face_image[12:-12, 12:-12]
    eye_lm_store = eye_lm_store * scale - (f_size[0]//2-43)
    #cv2.circle(face_image, (48, eye_lm_store), 5, (255, 0, 0), -1)
    # obtain left eye
    x_c = np.mean(leftEye_warped[0,[0,3]])
    x_c = x_c.astype(np.int_)
    y_c = np.mean(leftEye_warped[1,[0,3]])
    y_c = y_c.astype(np.int_)
    left_image = warped_image[y_c-80:y_c+80, x_c-80:x_c+80]
    M_left, s_left = point_to_matrix(leftEye_warped[:,[0,3]].transpose().reshape(-1), desiredDist = 96*0.7)

    left_image = cv2.warpAffine(left_image, M_left, (160,160))
    left_image = left_image[80-40-5:80+40-5, 80-60:80+60]
    left_image = grayNhist(left_image)
    left_image = left_image[8:-8, 12:-12]
    # obtain right eye
    x_c = np.mean(rigtEye_warped[0,[0,3]])
    x_c = x_c.astype(np.int_)
    y_c = np.mean(rigtEye_warped[1,[0,3]])
    y_c = y_c.astype(np.int_)
    rigt_image = warped_image[y_c-80:y_c+80, x_c-80:x_c+80]
    M_rigt, s_rigt = point_to_matrix(rigtEye_warped[:,[0,3]].transpose().reshape(-1), desiredDist = 96*0.7)
    rigt_image = cv2.warpAffine(rigt_image, M_rigt, (160,160))
    rigt_image = rigt_image[80-40-5:80+40-5, 80-60:80+60]
    rigt_image = grayNhist(rigt_image)
    rigt_image = rigt_image[8:-8, 12:-12]

    return face_image, left_image, rigt_image, eye_lm_store, fc_c_world / np.linalg.norm(fc_c_world), warpMat


def WarpNDraw(input_image, eye_lm, gaze_vec, cam_new, inv_cam_new):
    """ crop images by perspective warping """

    #eye_lm = 48
    output_image = np.array(input_image)
    eye_lm_2d = np.array([[48], [eye_lm], [1.]])
    eye_lm_3d = np.matmul(inv_cam_new, eye_lm_2d)

    g_end = np.matmul(cam_new, eye_lm_3d - 0.05*gaze_vec.transpose())
    #print(g_end[1,0])
    g_end = g_end[:2,0] / g_end[2,0]

    cv2.line(output_image, (48, int(eye_lm)), tuple(g_end), (0, 255, 0), 2)

    return output_image
