# -*- coding: utf-8 -*-
# @Time    : 18-8-16 下午4:47
# @Author  : zhangmr
# @File    : generator.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import gfile
import tensorflow as tf


class ImgGenerator:
    def __init__(self, base_dir, feature_extractor, is_train=True):
        self.base_dir = base_dir
        self.feature_extractor = feature_extractor
        self.attributes, self.classes = self.get_attrs(os.path.join(base_dir, 'attributes_per_class.txt'))

        if is_train:
            self.img_path = os.path.join(base_dir, 'train')
            self.label_path = os.path.join(base_dir, 'train.txt')
        else:
            self.img_path = os.path.join(base_dir, 'test')

    def split_data(self, split_size=0.2):
        """
        在训练时候使用它
        :param split_size: unseen class所占的比例
        :return: x_train, x_eval, y_train, y_eval, s_train, s_eval
        """
        train_classes, eval_classes = train_test_split(self.classes, test_size=split_size, random_state=42,
                                                       shuffle=True)

        x_train, y_train, s_train = self._get_data(train_classes)
        x_eval, y_eval, s_eval = self._get_data(eval_classes)

        # 将训练集中的20%混入验证集，这样验证集中既有seen classes, 也有unseen classes
        x_train, x_drop, y_train, y_drop, s_train, s_drop = train_test_split(x_train, y_train, s_train, test_size=0.2,
                                                                             random_state=9, shuffle=True)
        x_eval = np.concatenate((x_eval, x_drop), axis=0)
        y_eval = np.concatenate((y_eval, y_drop), axis=0)
        s_eval = np.concatenate((s_eval, s_drop), axis=0)

        # extract features
        return x_train, x_eval, y_train, y_eval, s_train, s_eval

    def _get_data(self, classes):
        imgs = []
        labels = []
        attributes = []

        with open(self.label_path) as f:
            pairs = f.readlines()

        for pair in pairs:
            img_name, img_label = pair.split()

            if img_label in classes:
                imgs.append(img_name)
                labels.append(img_label)
                id = list(self.classes).index(img_label)
                attributes.append(self.attributes[id])

        return np.array(imgs), np.array(labels), np.array(attributes)

    def get_imgs(self):
        """
        在预测test数据集时使用它
        :return:
        """
        imgs = os.listdir(self.img_path)
        imgs = [os.path.join(self.img_path, img) for img in imgs]
        return imgs

    def get_attrs(self, attributes_path):
        attributes = []
        classes = []
        with open(attributes_path) as f:
            per_class_attributes = f.readlines()

        for one_class_attribute in per_class_attributes:
            values = one_class_attribute.split()
            label = values.pop(0)
            values = [float(x) for x in values]
            attributes.append(values)
            classes.append(label)

        return np.array(attributes), np.array(classes)


class FeatureExtractor:
    def __init__(self, pb_dir):
        self.load_pb(pb_dir)

    def load_pb(self, pb_dir):
        output_graph_def = tf.GraphDef()
        with gfile.FastGFile(pb_dir, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

    def _get_feature(self, sess, img):
        encoder = sess.graph.get_tensor_by_name("encoder/MaxPool:0")
        features = sess.run(encoder, feed_dict={"x:0": img})
        return features

    def extract(self, imgs):
        features = []
        with tf.Session() as sess:
            for img in imgs:
                feature = self._get_feature(sess, [img])[0]
                features.append(feature)

        return np.array(features)
