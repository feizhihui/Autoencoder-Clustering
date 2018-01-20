# encoding=utf-8
import numpy as np

Psamples = np.load('../cache/Psamples.npy', )  # 十分之九正样本训练
MNsamples = np.load('../cache/MNsamples.npy', )  # 全部预测负样本测试

np.save('../cache/Psamples_demo.npy', Psamples[:10000])
np.save('../cache/MNsamples_demo.npy', MNsamples[:10000])
