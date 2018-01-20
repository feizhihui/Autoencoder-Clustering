# encoding=utf-8
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

learning_rate = 0.1
threshold = 0.5

height = 183
width = 234

# Store layers weight & bias  ==========================
weights = {
    # 5x5 conv, 3 input, 32 outputs 定义第一个卷积核
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs 定义第二个卷积核
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs 定义全连接权重
    'fn1': tf.Variable(tf.random_normal([8 * 10 * 32, 256], stddev=0.1)),
    # 定义第二层全连接层
    'fn2': tf.Variable(tf.random_normal([256, 1], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'fn1': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'fn2': tf.Variable(tf.random_normal([1], stddev=0.1))
}


class CNNModel(object):
    def __init__(self):
        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, height, width])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Construct model
        logits = self.conv_net(tf.expand_dims(self.X, -1), weights, biases, self.keep_prob)
        self.y_score = tf.sigmoid(logits)

        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=logits))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Evaluate model
        self.y_pred = tf.cast(tf.greater_equal(self.y_score, threshold), tf.int32)

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=5):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')  # SAME padding

    # Create model   #==============================================
    def conv_net(self, x, weights, biases, dropout):
        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        print(conv1)
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=5)
        print(conv1)
        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        print(conv2)
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=5)
        print(conv2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        out = tf.reshape(conv2, [-1, weights['fn1'].get_shape().as_list()[0]])
        out = tf.nn.dropout(out, dropout)
        # 第一层全连接层
        fc1 = tf.add(tf.matmul(out, weights['fn1']), biases['fn1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)
        # 第二层全连接层
        fn2 = tf.add(tf.matmul(fc1, weights['fn2']), biases['fn2'])

        return fn2


if __name__ == '__main__':
    print(CNNModel())
