# encoding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split


# from keras.utils import to_categorical


# CNsamples  demo
# Psamples MNsamples
def get_dataset():
    Psamples = np.load('../cache/Psamples_demo.npy')  # 十分之九正样本训练
    MNsamples = np.load('../cache/MNsamples_demo.npy')

    X = np.concatenate((Psamples, MNsamples), axis=0)
    X = np.expand_dims(X, axis=-1)
    Y = np.concatenate([np.ones([len(Psamples)]), np.zeros([len(MNsamples)])], axis=0)
    # Y = to_categorical(Y, 2)

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Add noise:
    # noise_factor = 0.1
    # X_train_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    # X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    #
    # X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    # X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    # return X_train_noisy, X_test_noisy, Y, Y_test
    return X, X_test, Y, Y_test
