# encoding=utf-8
import numpy as np
import pickle

train_eval_rate = 0.9


class DataLoader(object):
    def __init__(self, train_mode=True):
        if train_mode:
            Psamples = np.load('../cache/Psamples.npy', )  # 十分之九正样本训练
            # CNsamples = np.load('../cache/CNsamples.npy', )  # 全部确认负样本训练
            MNsamples = np.load('../cache/MNsamples.npy', )
            # Npresamples = np.load('../cache/Npresamples.npy', )  # 全部预测负样本训练

            self.pos_size = int(train_eval_rate * Psamples.shape[0])
            self.pos_train_X = Psamples[:self.pos_size]
            self.neg_size = MNsamples.shape[0]
            self.neg_train_X = MNsamples

            self.train_X = np.concatenate([self.pos_train_X, self.neg_train_X], axis=0)
            self.train_Y = np.concatenate([np.ones([self.pos_size, 1]), np.zeros([self.neg_size, 1])], axis=0)

            self.train_size = self.pos_size + self.neg_size

            print('positive and negative samples {:d},{:d}'.format(self.pos_size, self.neg_size))
        else:
            Psamples = np.load('../cache/Psamples.npy', )  # 十分之一正样本测试
            MNsamples = np.load('../cache/MNsamples.npy', )  # 全部预测负样本测试
            pos_num = Psamples.shape[0] - int(Psamples.shape[0] * train_eval_rate)
            neg_num = MNsamples.shape[0]
            self.test_X = np.concatenate([Psamples[-pos_num:], MNsamples], axis=0)
            self.test_Y = np.concatenate([np.ones([pos_num, 1]), np.zeros([neg_num, 1])], axis=0)
            self.test_size = len(self.test_X)
            print('testing size is', self.test_size)

    def shuffle(self):

        mark = list(range(self.train_size))
        np.random.shuffle(mark)
        self.train_X = self.train_X[mark]
        self.train_Y = self.train_Y[mark]


if __name__ == '__main__':
    DataLoader()
