# encoding=utf-8

import time

import xgboost as xgb
from sklearn.metrics import roc_auc_score

from autoencoder_cnn.get_dataset import get_dataset

tstart = time.clock()
train_X, test_X, train_Y, test_Y = get_dataset()

print('dataset loaded finished! ', time.clock() - tstart)
dtrain = xgb.DMatrix(train_X.reshape([-1, 183 * 234]), label=train_Y)
dtest = xgb.DMatrix(test_X.reshape([-1, 183 * 234]), label=test_Y)

param = {'max_depth': 6, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 300  # 200
param['nthread'] = 8
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
model = xgb.train(plst, dtrain, num_round, evallist)

sub_test_X = xgb.DMatrix(test_X.reshape([-1, 183 * 234]))
y_pred = model.predict(sub_test_X)

auc = roc_auc_score(test_Y, y_pred)
print(auc)
print(y_pred[:20])
print(test_Y[:20])
