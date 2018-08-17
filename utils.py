# -*- coding: utf-8 -*-
# @Time    : 18-8-16 下午4:47
# @Author  : zhangmr
# @File    : utils.py

from sklearn.metrics.pairwise import pairwise_distances_argmin
import numpy as np
import os


def nearest_neighbor(s_samples, s_classes):
    """
    :param s_samples: (n_samples, attributes_dim)
    :param s_classes: (n_classes, attributes_dim)
    :return: (n_samples, ), 返回每个测试样本在S空间距离最近的class index
    """
    min_dist_pos = pairwise_distances_argmin(s_samples, s_classes)
    return np.array(min_dist_pos)


def accuracy(y_pred, y_true):
    """
    :param y_pred: 预测标签
    :param y_true: 真实标签
    :return: 准确率
    """
    correct = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_pred)


def submit(imgs, predictions, filename):
    with open(filename, 'w') as f:
        for i, img in enumerate(imgs):
            img_name = img.split('/')[-1]
            line = img_name + "\t" + predictions[i] + "\r\n"
            f.write(line)
    print("Generate submit done!")
