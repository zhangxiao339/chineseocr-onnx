#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 5:53 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : text_Infer.py

import cv2
import onnxruntime as rt
import numpy as np


class TextRecHandle:
    def __init__(self, model_path, keywords_str):
        """
        初始化模型
        :param model_path: 模型路径
        :param keywords_str: words
        """
        self.keywords_str = keywords_str
        self.characters = keywords_str
        self.characters = self.characters + 'ç'
        self.key_size = len(keywords_str) + 1
        self.model_path = model_path
        self.is_inited = self.init()

    def decode(self, pred):
        char_list = []
        pred_text = pred.argmax(axis=3)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != 0 and (not(i > 0 and pred_text[i - 1] == pred_text[i])):
                char_list.append(self.characters[int(pred_text[i]) - 1])
        score = pred.max(-1).mean()
        return u''.join(char_list), score

    def init(self):
        try:
            self.sess = rt.InferenceSession(self.model_path)
            return True
        except Exception as e:
            print(e)
            return False
        
    def predict(self, mat):
        assert self.is_inited, 'the model not inited'
        image = mat.copy()
        if len(image.shape) > 2 and image.shape[2] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        newW = image.shape[1]
        if image.shape[0] != 32:
            newW = int((32 / image.shape[0]) * image.shape[1])
            image = cv2.resize(image, (newW, 32), cv2.INTER_LANCZOS4)  # image.resize((newW, 32), Image.BILINEAR)
        image = (image.astype(np.float32) / 255 - 0.5) / 0.5
        x = image.reshape([1, 1, 32, newW])
        preds = self.sess.run(["out"], {"input": x.astype(np.float32)})
        out, score = self.decode(np.asarray(preds))
        return out, score
