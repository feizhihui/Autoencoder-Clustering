# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

learning_rate = 0.01

height = 183
width = 234

hidden_num1 = 512
hidden_num2 = 256
hidden_num3 = 512


class AutoEncoder(object):
    def __init__(self):
        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, height, width])
        X = tf.reshape(self.X, [-1, height * width])

        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        hidden1 = layers.fully_connected(X, hidden_num1,
                                         weights_initializer=tf.truncated_normal_initializer(
                                             stddev=np.sqrt(2. / (height * width))),
                                         biases_initializer=tf.zeros_initializer(),
                                         activation_fn=tf.nn.relu)

        hidden2 = layers.fully_connected(hidden1, hidden_num2,
                                         weights_initializer=tf.truncated_normal_initializer(
                                             stddev=np.sqrt(2. / (hidden_num2))),
                                         biases_initializer=tf.zeros_initializer(),
                                         activation_fn=tf.nn.relu)

        hidden3 = layers.fully_connected(hidden2, hidden_num3,
                                         weights_initializer=tf.truncated_normal_initializer(
                                             stddev=np.sqrt(2. / (hidden_num2))),
                                         biases_initializer=tf.zeros_initializer(),
                                         activation_fn=tf.nn.relu)

        logits = layers.fully_connected(hidden3, height * width,
                                        weights_initializer=tf.truncated_normal_initializer(
                                            stddev=np.sqrt(2. / (hidden_num3))),
                                        biases_initializer=tf.zeros_initializer(),
                                        activation_fn=tf.nn.relu)
        self.loss = tf.reduce_mean(tf.square(logits - X)) * height * width

        # Define loss and optimizer
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=logits))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


if __name__ == '__main__':
    print(AutoEncoder())
