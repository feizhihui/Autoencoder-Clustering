# encoding=utf-8
import pandas as pd
import numpy as np
import time

start = time.clock()
CNsamples = pd.read_csv('../data/CNsamples.txt', sep='\t', header=None).values[:, :-1].reshape([-1, 183, 234])
print('CNsamples:', CNsamples.shape[0], time.clock() - start)
np.save('../cache/CNsamples.npy', CNsamples)
MNsamples = pd.read_csv('../data/MNsamples.txt', sep='\t', header=None).values[:, :-1].reshape([-1, 183, 234])
print('MNsamples:', MNsamples.shape[0], time.clock() - start)
np.save('../cache/MNsamples.npy', MNsamples)
Npresamples = pd.read_csv('../data/Npresamples.txt', sep='\t', header=None).values[:, :-1].reshape([-1, 183, 234])
print('Npresamples:', Npresamples.shape[0], time.clock() - start)
np.save('../cache/Npresamples.npy', Npresamples)
Psamples = pd.read_csv('../data/Psamples.txt', sep='\t', header=None).values[:, :-1].reshape([-1, 183, 234])
print('Psamples:', Psamples.shape[0], time.clock() - start)
np.save('../cache/Psamples.npy', Psamples)
