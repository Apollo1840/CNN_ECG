from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from biosppy.signals import ecg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers


def Conv1d(number_of_classes=4):
    model = Sequential()
    model.add(Conv1D(128, 55, activation='relu', input_shape=(big, 1)))
    model.add(MaxPooling1D(10))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 25, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 10, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # model.add(Flatten())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model