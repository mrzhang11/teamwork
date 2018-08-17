# -*- coding: utf-8 -*-
# @Time    : 18-8-16 下午4:47
# @Author  : zhangmr
# @File    : generator.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import gfile
import tensorflow as tf
import matplotlib.pyplot as plt


class ImgGenerator:
    def __init__(self, base_dir, feature_extractor, is_train=True):
        self.base_dir = base_dir
        self.feature_extractor = feature_extractor
        self.attributes, self.classes = self._get_maps(os.path.join(base_dir, 'attributes_per_class.txt'))

        print("attributes shape: ", self.attributes.shape)
        print("classes shape: ", self.classes.shape)

        if is_train:
            self.img_path = os.path.join(base_dir, 'train')
            self.label_path = os.path.join(base_dir, 'train.txt')
            self.x, self.y, self.s = self._get_data()
            print("extracting features----------------")
            self.xf = feature_extractor.extract(self.x)

        else:
            self.img_path = os.path.join(base_dir, 'test')
            self.x = self._get_imgs()
            print("extracting features----------------")
            self.xf = feature_extractor.extract(self.x)

    def _get_maps(self, attributes_path):
        """
        获取所有的类别及类别属性。
        :param attributes_path: attributes_per_class.txt所在路径
        :return: arrays
        """
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

    def _get_imgs(self):
        """
        获取图像目录下所有的图片名（绝对路径）
        :return:
        """
        imgs = os.listdir(self.img_path)
        imgs = [os.path.join(self.img_path, img) for img in imgs]
        return imgs

    def _get_data(self):
        """
        根据train.txt获取所有图片名（绝对路径）、类别标签、类别属性
        :return:
        """
        imgs = []
        labels = []
        attributes = []

        with open(self.label_path) as f:
            pairs = f.readlines()

        for pair in pairs:
            img_name, img_label = pair.split()
            imgs.append(os.path.join(self.img_path, img_name))
            labels.append(img_label)
            id = list(self.classes).index(img_label)
            attributes.append(self.attributes[id])

        return np.array(imgs), np.array(labels), np.array(attributes)

    def split_data(self, split_size=0.2):
        """
        划分训练集和验证集。
        :param split_size: unseen class所占的比例
        :return: xf_train, xf_eval, y_train, y_eval, s_train, s_eval
        """
        train_classes, eval_classes = train_test_split(self.classes, test_size=split_size, random_state=42,
                                                       shuffle=True)

        train_indices = self._get_indices(train_classes)
        eval_indices = self._get_indices(eval_classes)

        # 将训练集中的20%混入验证集，这样验证集中既有seen classes, 也有unseen classes
        train_indices, drop_indices = train_test_split(train_indices, test_size=split_size, random_state=9, shuffle=True)
        eval_indices.extend(drop_indices)

        x_train, x_eval, y_train, y_eval, s_train, s_eval = self.xf[train_indices], self.xf[eval_indices], \
                                                             self.y[train_indices], self.y[eval_indices], \
                                                                self.s[train_indices], self.s[eval_indices]

        print("training data shape:")
        print("feature: ", x_train.shape)
        print("label: ", y_train.shape)
        print("attributes: ", s_train.shape)

        print("evaluating data shape:")
        print("feature: ", x_eval.shape)
        print("label: ", y_eval.shape)
        print("attributes: ", s_eval.shape)

        return x_train, x_eval, y_train, y_eval, s_train, s_eval

    def _get_indices(self, classes):
        """
        根据所给类别，筛选得到这些类别对应的indices
        :param classes: 类别子集
        :return: 索引
        """
        with open(self.label_path) as f:
            pairs = f.readlines()
        indices = [id for id, pair in enumerate(pairs) if pair.split()[1] in classes]

        return indices


class FeatureExtractor:
    def __init__(self, pb_dir):
        self.sess = tf.Session()
        self._load_pb(pb_dir)

    def _load_pb(self, pb_dir):
        output_graph_def = tf.GraphDef()
        with gfile.FastGFile(pb_dir, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

    def _get_feature(self, img):
        img = plt.imread(img)
        img = img / 255
        if len(img.shape)==2:
            one_channel = img[:,:,np.newaxis]
            img = np.concatenate((one_channel, one_channel, one_channel), axis=-1)

        encoder = self.sess.graph.get_tensor_by_name("encoder/MaxPool:0")
        feature = self.sess.run(encoder, feed_dict={"x:0": [img]})
        feature = list(np.reshape(feature, [1, -1]))
        return feature

    def extract(self, imgs):
        features = []
        for img in imgs:
            feature = self._get_feature(img)[0]
            features.append(feature)

        return np.array(features)
