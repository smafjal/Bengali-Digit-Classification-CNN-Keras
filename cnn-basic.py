#!/usr/bin/env python

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import data_reader as reader

np.random.seed(1337)
batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def my_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model

def save_model(model,path):
    model.save(path)

def load_model(path):
    model=load_weights(path)
    return model

def train():
    X_train, Y_train, X_test, Y_test = get_data()
    input_shape=(img_rows,img_cols,1)
    model=my_model(input_shape) # get model
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))

    save_model(model,'pretrained/cnn-basic-model.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def get_data():
    data_x = reader.read_data("pickle/img_data.pickle")
    data_y = reader.read_data("pickle/img_label.pickle")

    tr_lim = int(len(data_x) * 70 / 100)

    X_train, Y_train = data_x[:tr_lim], data_y[:tr_lim]
    X_test, Y_test = data_x[tr_lim:], data_y[tr_lim:]

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, Y_train, X_test, Y_test

def print_data(data):
    for x in data:
        print len(x), len(x[0])
    print "==" * 50


def main():
    train()


if __name__ == "__main__":
    main()
