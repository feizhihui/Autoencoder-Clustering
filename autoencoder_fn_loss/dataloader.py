# encoding=utf-8
import numpy as np

train_eval_rate = 0.9


class DataLoader(object):
    def __init__(self, train_mode=True):
        if train_mode:
            Psamples = np.load('../cache/Psamples_demo.npy', )  # 十分之九正样本训练
            psize = int(Psamples.shape[0] * train_eval_rate)
            self.train_X = Psamples[:psize]
            self.train_size = psize
            print('training samples {:d}'.format(self.train_size))
        else:
            MNsamples = np.load('../cache/MNsamples_demo.npy', )  # 全部预测负样本测试
            self.test_X = MNsamples
            self.test_size = MNsamples.shape[0]
            print('testing size is', self.test_size)

    def shuffle(self):
        mark = list(range(self.train_size))
        np.random.shuffle(mark)
        self.train_X = self.train_X[mark]

    def transformer(self):
        Psamples = np.load('../cache/Psamples_demo.npy', )  # 十分之一正样本测试
        psize = int(Psamples.shape[0] * train_eval_rate)
        self.test_X = Psamples[psize:]
        self.test_size = Psamples.shape[0] - psize
        print('testing size is', self.test_size)


if __name__ == '__main__':
    DataLoader()
