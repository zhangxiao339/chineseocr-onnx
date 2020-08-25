# -*- coding: utf-8 -*-
# @Time : 2020/7/20 16:52
# @Author : ZhangXiao (sinceresky@foxmail.com, sinceresky520@gmail.com)
# @Site :  https://github.com/zhangxiao339
# @File : DBNetDetector.py
# @Software: PyCharm


import onnxruntime as rt
import numpy as np
import cv2
import time
from .DecodeUtil import DetectorDecoder

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=1):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)

        cv2.polylines(img_path, [point + 1], True, color, thickness)
    return img_path


def draw_label_points(img_path, label_list, color=(0, 255, 0), thickness=1):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for points in label_list:
        # points = points.astype(int)
        # cv2.drawContours(img_path, points, color=color, thickness=thickness)
        for point in points:
            img_path[point[1]][point[0]][0] = 0
            img_path[point[1]][point[0]][1] = 255
            img_path[point[1]][point[0]][2] = 0

    return img_path


class SingletonType(type):
    def __init__(cls, *args, **kwargs):
        super(SingletonType, cls).__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(obj, *args, **kwargs)
        return obj


class DBNETDetector:
    def __init__(self, config):
        self.config = config
        self.sess = rt.InferenceSession(self.config.detect_mod_path)
        self.short_size = self.config.text_detector_short_length
        self.decode_handel = DetectorDecoder(thresh=self.config.dbnet_pix_thresh,
                                             max_candidates =self.config.detect_max_box_size,
                                             box_thresh=self.config.dbnet_box_thresh,
                                             min_box_size=self.config.min_box_size)

    def process_data(self, img):
        h, w = img.shape[:2]
        if h < w:
            if h < self.short_size:
                tar_h = (int(h / 32) + 1) * 32
                tar_w = (int(w / 32) + 1) * 32
            else:
                tar_h = self.short_size
                tar_w = (tar_h / h) * w
                if tar_w % 32 != 0:
                    tar_w = (int(tar_w / 32) + 1) * 32
        else:
            if w < self.short_size:
                tar_w = (int(w / 32) + 1) * 32
                tar_h = (int(h / 32) + 1) * 32
            else:
                tar_w = self.short_size
                tar_h = (tar_w / w) * h
                if tar_h % 32 != 0:
                    tar_h = (int(tar_h / 32) + 1) * 32
        tar_img = cv2.resize(img, (tar_w, tar_h))
        return tar_img

    def process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if h < w:
            scale_h = self.short_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w

        else:
            scale_w = self.short_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h

        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)

        img = img.astype(np.float32)

        img /= 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess.run(["out1"], {"input0": transformed_image.astype(np.float32)})
        box_list, score_list= self.decode_handel(out[0][0], h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        return box_list, score_list


if __name__ == "__main__":
    text_handle = DBNETDetector(MODEL_PATH="../mod/text_detect.onnx", short_size=960)
    img = cv2.imread("../data/test/demo.png")
    start_time = time.time()
    box_list, score_list = text_handle.process(img)
    use_time = time.time() - start_time
    print('input size: {}, use time: {}'.format(img.shape, use_time))
    img1 = draw_bbox(img, box_list)
    cv2.imwrite("temp.jpg", img1)
    # img2 = draw_label_points(img, label_list)
    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * label_img
    # cv2.imwrite("baodan1_label.jpg", img2)
