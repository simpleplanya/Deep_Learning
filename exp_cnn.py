# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:05:29 2020

@author: Rocky
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np 
import pandas as pd


def map_size(x_train,x_test):    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)       
    return x_train,x_test



# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()



bool_idx = y_train <5 
x_train_remap = x_train[bool_idx]
y_train_remap = y_train[bool_idx]
bool_idx = y_test <5
x_test_remap = x_test[bool_idx]
y_test_remap = y_test[bool_idx]


batch_size = 128
num_classes = 5
epochs = 6

input_shape = (img_rows, img_cols, 1)
x_train_remap , x_test_remap = map_size(x_train_remap,x_test_remap)

x_train_remap = x_train_remap.astype('float32')
x_test_remap = x_test_remap.astype('float32')
x_train_remap /= 255
x_test_remap /= 255
print('x_train shape:', x_train.shape)
print(x_train_remap.shape[0], 'train samples')
print(x_test_remap.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train_remap = keras.utils.to_categorical(y_train_remap, num_classes)
y_test_remap = keras.utils.to_categorical(y_test_remap, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train_remap, y_train_remap,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_remap, y_test_remap))
score = model.evaluate(x_test_remap, y_test_remap, verbose=0)
pre_for_test_set = model.predict(x_test_remap)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

''' given other number test '''

bool_idx = y_test <5

x_inference_remap = x_test[bool_idx]
y_inference_remap = y_test[bool_idx]
x_inference_remap = x_inference_remap.reshape(x_inference_remap.shape[0], img_rows, img_cols, 1)
# if not translate data type , what happen?
x_test_remap.astype('float32')
x_test_remap/=255
res = model.predict(x_inference_remap)

print('predict acc ')
decition = np.argmax(res,axis=1)
max_pro = np.max(res,axis=1)
inference_ans1= pd.DataFrame({'label':y_inference_remap,'decition':decition,'max_pro':max_pro})
inference_ans1

bool_idx = y_test >=5
x_inference_remap = x_test[bool_idx]
y_inference_remap = y_test[bool_idx]
x_inference_remap = x_inference_remap.reshape(x_inference_remap.shape[0], img_rows, img_cols, 1)
# if not translate data type , what happen?
x_test_remap.astype('float32')
x_test_remap/=255
res = model.predict(x_inference_remap)

#print('predict acc ')
decition = np.argmax(res,axis=1)
max_pro = np.max(res,axis=1)
inference_ans2= pd.DataFrame({'label':y_inference_remap,'decition':decition,'max_pro':max_pro})
inference_ans2



