# -*- coding: utf-8 -*-
# @Time    : 18-8-10 下午12:17
# @Author  : zhangmr
# @File    : data_loader.py

import tensorflow as tf
import os
from numpy.random import permutation


class ImgLoader:
    def __init__(self, data_dir, data_shape=(64, 64, 3), batch_size=64, is_shuffle=True, is_repeat=True):
        self.data_dir = data_dir
        self.data_shape = data_shape
        self.iterator = tf.data.Iterator.from_structure(tf.float32, ([None, data_shape[0], data_shape[1], data_shape[2]]))
        self.dataset = self.__create_dataset(batch_size, is_shuffle, is_repeat)
        self.element = None

    def __create_dataset(self, batch_size, is_shuffle, is_repeat):
        # get image files
        self.img = os.listdir(self.data_dir)
        self.img_path = [os.path.join(self.data_dir, file) for file in self.img]
        # get dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.img_path)
        dataset = dataset.map(self.__parse, num_parallel_calls=4)
        if is_shuffle:
            dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
        if is_repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset

    def __parse(self, img_path):
        content = tf.read_file(img_path)
        img = tf.image.decode_png(content, channels=3)
        img = img/255
        return img

    def initialize(self, sess):
        training_init_op = self.iterator.make_initializer(self.dataset)
        sess.run(training_init_op)
        self.element = self.iterator.get_next()

    def next_batch(self):
        return self.element


class SplitClassDataLoader:
    def __init__(self, split=0.2, attributes_path='DatasetA/attributes_per_class.txt',
                 img_label_path='DatasetA/train.txt', classes_path='DatasetA/label_list.txt'):
        """
        Split data. After spliting, training dataset is disjoint with evaluating dataset.
        Currently, don't use test data.
        :param split: split the original dataset into (1-split)training data and (split)evaluating data.
        :param img_dir: images directory
        :param attributes_dir: classes and their attributes(manual annotation)
        :param label_dir:  image names and their label
        """
        self._split = split
        # categorical
        self.total_classes = self.get_total_classes(classes_path)
        self.seen_classes = self.get_seen_classes(img_label_path)
        # numerical
        self.train_classes, self.eval_classes = self.split_classes(self.seen_classes)

        self.train_imgs, self.train_labels = self.get_data_with_classes(img_label_path, self.train_classes)
        self.eval_imgs, self.eval_labels = self.get_data_with_classes(img_label_path, self.eval_classes)

        self.attributes_map = self.get_attributes_map(attributes_path)

    @staticmethod
    def get_total_classes(classes_path):
        """
        :param classes_path:
        :return: classes, length:190
        """
        classes = dict()
        with open(classes_path) as f:
            class_pairs = f.readlines()
        for pair in class_pairs:
            class_label, class_name = pair.split()
            classes[class_label] = class_name

        return classes

    @staticmethod
    def get_seen_classes(img_label_path):
        """
        :param img_label_path:
        :return: seen classes,length:149
        """
        classes = set()
        with open(img_label_path) as f:
            class_pairs = f.readlines()
        for pair in class_pairs:
            img_name, img_label = pair.split()
            classes.add(img_label)

        return list(classes)

    def split_classes(self, classes):
        random_classes = permutation(classes)
        seg_index = int(len(classes)*self._split)
        eval_classes = random_classes[: seg_index]
        train_classes = random_classes[seg_index: -1]

        return list(train_classes), list(eval_classes)

    def get_data_with_classes(self, img_label_path, classes):
        imgs = []
        labels = []
        with open(img_label_path) as f:
            pairs = f.readlines()
        for pair in pairs:
            img_name, img_label_categorical = pair.split()

            if img_label_categorical in classes:
                imgs.append(img_name)
                labels.append(img_label_categorical)

        return imgs, labels

    def get_attributes_map(self, attributes_path):
        attributes = dict()
        with open(attributes_path) as f:
            per_class_attributes = f.readlines()
        for one_class_attribute in per_class_attributes:
            values = one_class_attribute.split()
            label = values.pop(0)
            values = [float(x) for x in values]
            attributes[label] = values

        return attributes


