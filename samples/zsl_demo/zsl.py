# -*- coding: utf-8 -*-
# @Time    : 18-8-10 下午4:47
# @Author  : zhangmr
# @File    : zsl.py

import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import scipy as sp
import numpy as np

from .data_loader import ImgLoader
from .feature import FeatureExtractor


class ZSL:
    def __init__(self, exp_dataloader, parameters):
        self.exp_dataloader = exp_dataloader
        self._parameters = parameters

    def __read_x(self, data_dir, model_dir):
        print("Extracting X features...")

        img_loader = ImgLoader(data_dir=data_dir, is_shuffle=False, is_repeat=False)
        extractor = FeatureExtractor(img_loader, pb_dir=model_dir)
        df = extractor.extract()

        sub_df1 = df[df["name"].isin(self.exp_dataloader.train_imgs)]
        sub_df2 = pd.DataFrame({"name": self.exp_dataloader.train_imgs})
        new_df = pd.merge(sub_df2, sub_df1, on='name')
        x_train = np.array(new_df['features'].tolist())

        sub_df1 = df[df["name"].isin(self.exp_dataloader.eval_imgs)]
        sub_df2 = pd.DataFrame({"name": self.exp_dataloader.eval_imgs})
        new_df = pd.merge(sub_df2, sub_df1, on='name')
        x_eval = np.array(new_df['features'].tolist())

        return x_train, x_eval

    def __read_s(self):
        print("reading attributes...")
        attributes = self.exp_dataloader.attributes_map

        train_classes = self.exp_dataloader.train_classes
        eval_classes = self.exp_dataloader.eval_classes
        s_train = np.array([v for k, v in attributes.items() if k in train_classes])
        s_eval = np.array([v for k, v in attributes.items() if k in eval_classes])

        return s_train, s_eval

    @staticmethod
    def __create_gaussian_kernel(x, z, sigma):
        pairwise_dists = pairwise_distances(x, z, metric='euclidean', n_jobs=-1)
        k = sp.exp(-pairwise_dists ** 2 / sigma ** 2)
        return k

    @staticmethod
    def __create_linear_kernel(x, z):
        k = np.dot(x, z.T)
        return k

    def __calculate_y(self, labels, classes):
        """
        :param labels: categorical_labels
        :param classes: classes array
        :return: one-hot encoded Y
        """
        y = np.zeros((len(labels), len(classes)))

        for idx in range(len(labels)):
            y[idx, classes.index(labels[idx])] = 1

        return y

    def __calculate_k(self, x, z):
        print("calculating K...")

        if self._parameters['sigma'] == 0:
            k = self.__create_linear_kernel(x, z)
        else:
            k = self.__create_gaussian_kernel(x, z, self._parameters['sigma'])

        return k

    def __calculate_v(self, k, y, s):
        kk = k.T @ k
        print("KK:", kk.shape)

        kys = k @ y @ s
        print("K*Y*S:", kys.shape)

        kys_inv_ss = kys @ sp.linalg.inv(s.T @ s + self._parameters['lambda'] * np.identity(np.shape(s)[1]))
        v = sp.linalg.inv(kk + self._parameters['gamma'] * np.identity(np.shape(k)[1])) @ kys_inv_ss
        return v

    def __predict_classes(self, k, v, s):
        prediction_matrix = k @ v @ s.T

        class_prediction = np.argmax(prediction_matrix, axis=1)
        return class_prediction

    @staticmethod
    def __evaluate(labels, predictions):
        correct = 0
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct += 1
        return correct / len(predictions)

    def run(self):
        x_train, x_test = self.__read_x(self._parameters['data_dir'], self._parameters['model_dir'])
        print("X_train shape:", x_train.shape)
        print("X_test shape:", x_test.shape)

        y_train = self.__calculate_y(self.exp_dataloader.train_labels, self.exp_dataloader.train_classes)
        y_test = self.__calculate_y(self.exp_dataloader.eval_labels, self.exp_dataloader.eval_classes)
        print("Y_train shape:", y_train.shape)
        print("Y_test shape:", y_test.shape)

        k_train = self.__calculate_k(x_train, x_train)
        k_test = self.__calculate_k(x_test, x_train)
        print("K_train shape:", k_train.shape)
        print("K_test shape:", k_test.shape)

        s_train, s_test = self.__read_s()
        print("S_train shape:", s_train.shape)
        print("S_test shape:", s_test.shape)

        print("caculating V.....")
        v = self.__calculate_v(k_train, y_train, s_train)
        print("v shape(should be m*30):", v.shape)

        # Calculate class prediction of test set
        predictions = self.__predict_classes(k_test, v, s_test)
        print("predictions shape:", predictions.shape)
        np.savetxt("output/pred.txt", predictions, fmt='%d', newline='\n')

        labels = np.argmax(y_test, axis=1)
        np.savetxt("output/label.txt", labels, fmt='%d', newline='\n')
        np.savetxt("output/eval_class.txt", self.exp_dataloader.eval_classes, fmt='%s', newline='\n')

        # Calculate accuracy
        accuracy = self.__evaluate(labels, predictions)
        print("Final Score: %f" % accuracy)
