#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: voyager
# Date: 20190427
# Function: The program is programmed for WriterRecogition(HalfDeepWriter)

import os
import time
import random
import numpy as np
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import CSVLogger
from scipy.misc import imread, imresize

trn_list = "./trn/"
val_list = "./val/"
model_name = "model1.h5"
log_name = "log1.csv"

# generate a batch train data
def HWDB_load(set_list):
    while True:
        xs = []
        ys = []

        # choose a dir
        new_list = random.sample(os.listdir(set_list), 1)[0]
        for filename in os.listdir(set_list + new_list):
            # img
            img = imread(set_list + new_list + '/' + filename)
            img = img/255
            # label
            img_label = np.zeros(300)
            img_label[(int(filename[0:4]) - 1001)] = 1
            xs.append(img)
            ys.append(img_label)

        xs = np.reshape(xs, [-1, 113, 113, 1])
        ys = np.reshape(ys, [-1, 300])
        yield (xs, ys)


if __name__ == '__main__':
    start_time = time.time()

    model = Sequential()
    #conv1(96C5S2) MP1(M3S2)
    model.add(Conv2D(96, (5, 5), activation='relu', strides=(2, 2), padding='VALID', input_shape=(113, 113, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    #conv2(256C3S1P1) MP2(M3S2)
    model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    #conv3(384C3S1P1) conv4(384C3S1P1) conv5(256C3S1P1) MP3(M3S2)
    model.add(Conv2D(384, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    #fc1
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    #fc2
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    #softmax
    model.add(Dense(300, activation='softmax'))

    model.summary()
    csv_logger = CSVLogger(log_name, append=True, separator=';')
    ada = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(HWDB_load(trn_list), steps_per_epoch=1, epochs=30000, 
                                  validation_data=HWDB_load(val_list), validation_steps=1, callbacks=[csv_logger])

    #save the model
    model.save(model_name)
    end_time = time.time()
    print("\nTrain completed. Time cosuming:", end_time-start_time, 
          "\nModel is saved in", model_name ,
          "\nloss and accuracy values are saved in", log_name)

