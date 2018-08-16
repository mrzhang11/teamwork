# -*- coding: utf-8 -*-
# @Time    : 18-8-10 下午3:06
# @Author  : zhangmr
# @File    : feature.py

from tensorflow.python.platform import gfile
import tensorflow as tf
import pandas as pd
import numpy as np


class FeatureExtractor:
    def __init__(self, data_loader, pb_dir):
        self.data_loader = data_loader
        self.load_pb(pb_dir)

    @staticmethod
    def load_pb(pb_dir):
        output_graph_def = tf.GraphDef()
        with gfile.FastGFile(pb_dir, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

    @staticmethod
    def predict(sess, img):
        encoder = sess.graph.get_tensor_by_name("encoder/MaxPool:0")
        features = sess.run(encoder, feed_dict={"x:0": img})
        return features

    def save(self, img, feature, filename):
        pass

    def extract(self):
        imgs = self.data_loader.img
        features = []
        with tf.Session() as sess:
            self.data_loader.initialize(sess)
            try:
                cnt = 1
                while True:
                    x_batch = sess.run(self.data_loader.next_batch())
                    feature = self.predict(sess, x_batch)
                    # flatten features
                    feature = list(np.reshape(feature, [len(x_batch), -1]))
                    features.extend(feature)
                    cnt = cnt + 1
            except tf.errors.OutOfRangeError:
                print('Extract features done.')

        return pd.DataFrame({"name": imgs, "features": features})




