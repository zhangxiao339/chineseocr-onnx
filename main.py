#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:18 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : main.py

from config import chineseocr_config
from component.detector.DBNetDetector import DBNETDetector
from component.text_recongnizer.Text_Infer import TextRecHandle
import numpy as np
import cv2
import time

config = chineseocr_config()

detector_api = DBNETDetector(config)
text_rec_api = TextRecHandle(config.chineseocr_mod_file, config.chineseocr_words_str)


def get_rotate_crop_image(img, points):
    points += 0.3
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom + 1, left:right + 1, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1])) + 1
    img_crop_height = int(np.linalg.norm(points[0] - points[3])) + 1
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE)
    angle = 0
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
        angle = 90
    return dst_img, angle


def text_rec_with_box(mat, boxes_list, text_handle):
    results = []
    scores = []
    boxes_list = np.flipud(boxes_list)
    for index, box in enumerate(boxes_list):
        tmp_box = box.copy()
        tmp_box = tmp_box.astype(np.float32)
        part_mat, pre_rotate_angle = get_rotate_crop_image(mat, tmp_box)
        txt, score = text_handle.predict(part_mat)
        results.append(txt)
        scores.append(score)
    return results, scores


def text_rec(img):
    boxes_list, score_list = detector_api.process(img.copy())
    return text_rec_with_box(img.copy(), boxes_list, text_rec_api)


if __name__ == '__main__':
    pic = 'data/test/demo.png'
    mat = cv2.imread(pic)
    start = time.time()
    results, scores = text_rec(mat)
    t = time.time() - start
    for i in range(len(results)):
        print("line: {}, {}, {}".format(i + 1, results[i], scores[i]))
    print('total time: {}'.format(t))
