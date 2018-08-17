# -*- coding: utf-8 -*-
# @Time    : 18-8-16 下午4:48
# @Author  : zhangmr
# @File    : SAE.py
import os
from scipy.linalg import solve_sylvester
import numpy as np

from utils import *


class SAE:
    """
    Paper: Semantic Autoencoder for Zero-Shot Learning
    """

    def __init__(self, data_generator, params, load_weights=False):
        self.data_generator = data_generator
        self.params = params
        self.w = None
        if load_weights:
            self.load_weights()

    def load_weights(self):
        pwd = os.getcwd()
        filename = os.path.join(pwd, 'resource', 'SAE_weights.txt')
        self.w = np.loadtxt(filename)
        print("Load weights done! <%s>" % filename)

    def save_weights(self):
        pwd = os.getcwd()
        filename = os.path.join(pwd, 'resource', 'SAE_weights.txt')
        np.savetxt(filename, self.w, fmt='%.6f')
        print("Save weights done! <%s>" % filename)


    def _calculate_w(self, s, x, lambda_val):
        """
        :param s: array, (n, k)
        :param x:  array, (n, d)
        :param lambda_val: hyper-parameter
        :return: w: array, (k, d)
        """
        A = s.T @ s  # k*k
        B = lambda_val * x.T @ x  # d*d
        C = (1 + lambda_val) * s.T @ x  # k*d
        w = solve_sylvester(A, B, C)  # k*d

        return w

    def train(self):
        best_acc = 0
        best_lambda_val = 0

        print("start  training----------------")
        for lambda_val in self.params['lambdas']:
            acc = 0
            steps = 5
            for step in range(steps):
                x_train, x_eval, y_train, y_eval, s_train, s_eval = self.data_generator.split_data(split_size=0.2)
                self.w = self._calculate_w(s_train, x_train, lambda_val)
                _, cur_acc = self.predict(x_eval, y_eval)
                acc += cur_acc

                print("lambda: %f, step: %d, accuracy(eval): %.4f" % (lambda_val, step, cur_acc) )
            acc /= steps

            if acc > best_acc:
                best_acc = acc
                best_lambda_val = lambda_val

        print("After training, best lambda: %f, best accuracy(eval): %f" % (best_lambda_val, best_acc))

        x_train, x_eval, y_train, y_eval, s_train, s_eval = self.data_generator.split_data(split_size=0)
        self.w = self._calculate_w(s_train, x_train, best_lambda_val)
        self.save_weights()

    def predict(self, x, y=None):
        print("start predicting--------------------")

        s_pred = x @ self.w.T
        y_indices = nearest_neighbor(s_pred, self.data_generator.attributes)
        y_pred = self.data_generator.classes[y_indices]
        acc = accuracy(y_pred, y) if y is not None else None

        return y_pred, acc

