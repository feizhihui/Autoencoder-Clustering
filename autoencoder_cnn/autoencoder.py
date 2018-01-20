# coding: utf-8


import os

import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Activation, Lambda, Flatten, concatenate, \
    Reshape
from keras.models import Model

from autoencoder_cnn.get_dataset import get_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

X_train, X_test, Y_train, Y_test = get_dataset()

# About Dataset:
img_size = X_train.shape[1]
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,', X_train.shape[1], 'x', X_train.shape[2], 'size RGB image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,', X_test.shape[1], 'x', X_test.shape[2], 'size RGB image.\n')

num_class = 2


# Custom classifier function:
def classifier_func(x):
    return x - x + K.one_hot(K.argmax(x, axis=1), num_classes=num_class)  # trick


input_img = Input(shape=(180, 225, 1))
# crop_img = Cropping2D(cropping=((0, 3), (0, 9)), data_format=None)(input_img)  # (180,225)

layer_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
layer_1 = MaxPooling2D((3, 3))(layer_1)

layer_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_1)
layer_2 = MaxPooling2D((3, 3))(layer_2)

layer_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_2)
layer_3 = MaxPooling2D((5, 5))(layer_3)

flat_1 = Flatten()(layer_3)

fc_1 = Dense(256)(flat_1)
fc_1 = Activation('relu')(fc_1)

fc_2 = Dense(64)(fc_1)
fc_2 = Activation('relu')(fc_2)

fc_3 = Dense(num_class)(fc_2)
act_class = Lambda(classifier_func, output_shape=(num_class,))(fc_3)

# Merge Layers:

merge_1 = concatenate([act_class, fc_2])

# Decoder:
fc_4 = Dense(256)(merge_1)
fc_4 = Activation('relu')(fc_4)

fc_5 = Dense(1280)(fc_4)
fc_5 = Activation('relu')(fc_5)
print(fc_5)
reshape_1 = Reshape((4, 5, 64))(fc_5)  #

layer_4 = UpSampling2D((5, 5))(reshape_1)
layer_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_4)

layer_5 = UpSampling2D((3, 3))(layer_4)
layer_5 = Conv2D(64, (3, 3), activation='relu', padding='same')(layer_5)

layer_6 = UpSampling2D((3, 3))(layer_5)
layer_6 = Conv2D(32, (3, 3), activation='relu', padding='same')(layer_6)

layer_7 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer_6)

autoencoder = Model(input_img, layer_7)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

# Training Model:
epochs = 2
batch_size = 128
autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, X_test), shuffle=True)

decoded_imgs = autoencoder.predict(X_test)

# Split autoencoder:
encoder = Model(input_img, act_class)
encoder.summary()

# #### Use the code to finding which cluster:
# `np.argmax(<encoder_output>, axis=0)`


encode = encoder.predict(X_train)

class_dict = np.zeros((num_class, num_class))
for i, sample in enumerate(Y_train):
    class_dict[np.argmax(encode[i], axis=0)][np.argmax(sample)] += 1

print(class_dict)

neuron_class = np.zeros((num_class))
for i in range(num_class):
    neuron_class[i] = np.argmax(class_dict[i], axis=0)

print(neuron_class)  # 聚类id=>类别id


# Getting class as string:
def cat_dog(model_output):
    if model_output == 0:
        return "Negative"
    else:
        return "Positive"


encode = encoder.predict(X_test)

predicted = np.argmax(encode, axis=1)
for i, sample in enumerate(predicted):
    predicted[i] = neuron_class[predicted[i]]

comparison = predicted == np.argmax(Y_test, axis=1)
loss = 1 - np.sum(comparison.astype(int)) / Y_test.shape[0]  # 1-Accuracy

print('Loss:', loss)
print('Examples:')
for i in range(20):
    neuron = np.argmax(encode[i], axis=0)
    print('Class:', cat_dog(np.argmax(Y_test[i], axis=0)), '- Model\'s Output Class:', cat_dog(neuron_class[neuron]),
          '\n' * 2, '-' * 40)
