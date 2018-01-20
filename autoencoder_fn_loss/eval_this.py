# encoding=utf-8


import tensorflow as tf
from dataloader import DataLoader
from autoencoder import AutoEncoder
import sklearn.metrics  as metrics
import numpy as np
import os

# LD_LIBRARY_PATH   	/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def eval_print(y_true, y_pred, y_logits):
    print("accuracy %.6f" % metrics.accuracy_score(y_true, y_pred))
    print("Precision %.6f" % metrics.precision_score(y_true, y_pred))
    print("Recall %.6f" % metrics.recall_score(y_true, y_pred))
    print("f1_score %.6f" % metrics.f1_score(y_true, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_logits)
    print("auc_socre %.6f" % metrics.auc(fpr, tpr))


batch_size = 512

loader = DataLoader(train_mode=False)
model = AutoEncoder()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # store
    saver = tf.train.Saver()
    saver.restore(sess, '../model/ae_model')

    print('begin testing:')

    losses = []
    for iter, indices in enumerate(range(0, loader.test_size, batch_size)):  #
        batch_X = loader.test_X[indices:indices + batch_size]  #
        loss = sess.run(model.loss,
                        feed_dict={model.X: batch_X, model.keep_prob: 1.0})
        losses.append(loss)
        if iter % 10 == 0:
            print("===Testing Iter:{:d}/{:d}===".format(iter + 1, loader.test_size // batch_size))  #
            print('loss: %.3f' % (sum(losses) / len(losses)))

    print('average eval loss:', sum(losses) / len(losses))
    # ============================================================================
    loader.transformer()
    losses = []
    for iter, indices in enumerate(range(0, loader.test_size, batch_size)):  #
        batch_X = loader.test_X[indices:indices + batch_size]  #
        loss = sess.run(model.loss,
                        feed_dict={model.X: batch_X, model.keep_prob: 1.0})
        losses.append(loss)
        if iter % 10 == 0:
            print("===Testing Iter:{:d}/{:d}===".format(iter + 1, loader.test_size // batch_size))  #
            print('loss: %.3f' % (sum(losses) / len(losses)))

    print('average eval loss:', sum(losses) / len(losses))
