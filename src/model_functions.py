
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import keras
import sys
import tensorflow as tf
import pandas as pd
from keras import Sequential, Model
from keras import layers
from os import path
from keras.saving import load_model 
import argparse
from keras.utils import to_categorical
from keras import optimizers
from sklearn.utils import shuffle
from collections import Counter
from keras import metrics
from sklearn.model_selection import KFold
import sklearn.preprocessing as prep
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
import sklearn.model_selection as mdselect
import keras.applications as apl
import keras.regularizers as reg
import joblib
import tensorflow.python.keras.backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
import keras

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

def convert_bytes(byte_size):
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = byte_size
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def convert_minutes(total_minutes):
    total_seconds = total_minutes * 60
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def show_params(model: Model, name: str):
    print(f"Model {name}:")
    params = model.count_params()
    print(" | Total params:", "{:,}".format(params).replace(",", " "))
    print(" | Size        :", convert_bytes(params * 4))

def get_all_layer_outputs(model: Model):
    return K.function([model.layers[0].input],
                      [l.output for l in model.layers[1:]])
    
def get_last_layer_outputs(model: Model):
    return K.function([model.layers[0].input],
                      [model.layers[-2]])

def block2D(inp, filters: int, down_sample: bool = False):
    shorcut = inp
    strides = [2, 1] if down_sample else [1, 1]
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[0], padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[1], padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    if down_sample:
        shorcut = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=2, padding="same")(shorcut)
        shorcut = layers.BatchNormalization()(shorcut)
    
    x = layers.Add()([x, shorcut])
    x = layers.Activation("relu")(x)
    return x

def block1D(inp, filters: int, down_sample: bool = False):
    shorcut = inp
    strides = [2, 1] if down_sample else [1, 1]
    x = layers.Conv1D(filters=filters, kernel_size=3, strides=strides[0], padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters=filters, kernel_size=3, strides=strides[1], padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    if down_sample:
        shorcut = layers.Conv1D(filters=filters, kernel_size=3, strides=2, padding="same")(shorcut)
        shorcut = layers.BatchNormalization()(shorcut)
    
    x = layers.Add()([x, shorcut])
    x = layers.Activation("relu")(x)
    return x
