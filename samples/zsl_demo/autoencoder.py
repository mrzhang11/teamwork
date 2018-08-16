# -*- coding: utf-8 -*-
# @Time    : 18-8-10 上午10:41
# @Author  : zhangmr
# @File    : autoencoder.py

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import os

from .data_loader import ImgLoader

tf.logging.set_verbosity(tf.logging.INFO)


class AutoEncoder:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        # train parameters
        self.learning_rate = 1e-4
        self.num_steps = 1e5
        # train operations
        self.input_shape = None
        self.x = None
        self.encoded = None
        self.decoded = None
        self.train_ops = None

    def build_model(self):
        self.input_shape = [None, self.data_loader.data_shape[0], self.data_loader.data_shape[1], self.dataloader.data_shape[2]]
        self.x = tf.placeholder(tf.float32, shape=self.input_shape, name='x')
        # encoder
        # 64*64*3
        conv1 = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        # 64*64*16
        pooling1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, name='pooling1')
        # 32*32*16
        conv2 = tf.layers.conv2d(inputs=pooling1, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        # 32*32*16
        pooling2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, name='pooling2')
        # 16*16*16
        conv3 = tf.layers.conv2d(inputs=pooling2, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        # 16*16*16
        self.encoded = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, name='encoder')

        # decoder
        # 8*8*16
        upsample1 = tf.image.resize_images(self.encoded, size=(16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 16*16*16
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        # 16*16*16
        upsample2 = tf.image.resize_images(conv4, size=(32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 32*32*16
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        # 32*32*16
        upsample3 = tf.image.resize_images(conv5, size=(64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 64*64*16
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu)
        # 64*64*16
        self.decoded = tf.layers.conv2d(inputs=conv6, filters=3, kernel_size=3, padding='same', activation=None, name='decoder')
        # 64*64*3

        variables = tf.trainable_variables()
        l2_loss = tf.reduce_mean([tf.nn.l2_loss(var) for var in variables if 'bias' not in var.name])
        loss_op = tf.reduce_mean(tf.pow(self.x-self.decoded, 2)) + 0.0001*l2_loss
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)

        tf.summary.scalar("loss", loss_op)
        tf.summary.image("origin", self.x)
        tf.summary.image("reconstruct", self.decoded)
        summaries = tf.summary.merge_all()
        self.train_ops = [optimizer, summaries, loss_op]

        return self.train_ops

    def train(self):

        train_ops = self.build_model()
        init = tf.global_variables_initializer()

        # Start Training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            self.data_loader.initialize(sess)

            summary_writer = tf.summary.FileWriter("exp/experiment1/", graph=sess.graph)
            saver = tf.train.Saver(max_to_keep=5)

            # Training
            for i in range(1, int(self.num_steps + 1)):
                # Prepare Data
                x_batch = sess.run(self.data_loader.next_batch())
                _, _summaries, loss = sess.run(train_ops, feed_dict={self.x: x_batch})
                summary_writer.add_summary(_summaries, i)

                if i % 100 == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, loss))
                    saver.save(sess, "exp/experiment1/", i)


def export(checkpoint_dir, pb_dir):
    def write_pb_file(sess, output_node):
        # Returns a serialized `GraphDef` representation of this graph.
        input_graph_def = sess.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node)
        with gfile.FastGFile(pb_dir, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    def load_checkpoints_and_write_pb():
        """
        Load checkpoints and write pb.
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # 读取保存的模型数据
            saver = tf.train.Saver()
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                saver.restore(sess, latest_checkpoint)
                print("Checkpoints loaded")
            # 导出pb文件
            model_dir = os.path.join(pb_dir, 'encoder.pb')
            write_pb_file(sess, ["x", "encoder/MaxPool"], model_dir)
        print("done")

    data_loader = ImgLoader(data_dir='DatasetA/train', batch_size=128)
    auto_encoder = AutoEncoder(data_loader)
    auto_encoder.build_model()
    # save to pb
    load_checkpoints_and_write_pb(checkpoint_dir, pb_dir)


if __name__=='__main__':
    # train encoder
    # data_loader = ImgLoader(data_dir='DatasetA/train', batch_size=128)
    # auto_encoder = AutoEncoder(data_loader)
    # auto_encoder.train()

    # export model
    export('exp/experiment1', 'output/')