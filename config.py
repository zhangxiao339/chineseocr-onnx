#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:18 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : config.py
import os
filt_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(filt_path) + os.path.sep + ".")


def get_keyword_str(file):
    key_str = u''
    with open(file, 'r', encoding='utf8') as fin:
        for line in fin:
            key_str += line.replace('\n', '')
        fin.close()
    return key_str


class chineseocr_config(object):
    def __init__(self):
        self.mod_dir = os.path.join(father_path, 'mod')
        self.chineseocr_mod_file = os.path.join(self.mod_dir, 'chineseocr_densectc.onnx')
        self.chineseocr_words_file = os.path.join(self.mod_dir, 'keywords_chinese.dict')
        self.chineseocr_words_str = get_keyword_str(self.chineseocr_words_file)
        self.detect_mod_path = os.path.join(self.mod_dir, 'text_detect_dbnet.onnx')
        self.text_detector_short_length = 960  # 32times
        self.dbnet_pix_thresh = 0.2
        self.dbnet_box_thresh = 0.5
        self.detect_max_box_size = 1000
        self.min_box_size = 3