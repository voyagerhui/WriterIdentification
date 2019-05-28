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
BATCH_SIZE = 256
model_name = "model1.h5"
log_name = "log2.csv"

# generate a batch train data
def HWDB_load(set_list):
    while True:
        xs = []
        ys = []

        new_list = random.sample(os.listdir(set_list), BATCH_SIZE)
        for filename in new_list:
            # img
            img = imread(set_list + filename)
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

    model = load_model(model_name)
    model.summary()
    csv_logger = CSVLogger(log_name, append=True, separator=';')
    history = model.fit_generator(HWDB_load(trn_list), steps_per_epoch=1, epochs=15000,
                                  validation_data=HWDB_load(val_list), validation_steps=1, callbacks=[csv_logger])

    #save the model
    model.save(model_name)
    end_time = time.time()
    print("\nTrain completed. Time cosuming:", end_time-start_time, 
          "\nModel is saved in", model_name ,
          "\nloss and accuracy values are saved in", log_name)

