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
from typing import List, Tuple

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
else:
    print("No GPU detected. Using CPU.")

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

def convert_bytes(byte_size) -> str:
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = byte_size
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def convert_minutes(total_minutes) -> str:
    total_seconds = total_minutes * 60
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def show_params(model: Model, name: str):
    print(f"Model {name}:")
    params = model.count_params()
    print(" | Total params :", "{:,}".format(params).replace(",", " "))
    print(" | Size         :", convert_bytes(params * 4))

def get_all_layer_outputs(model: Model):
    return K.function([model.layers[0].input],
                      [l.output for l in model.layers[1:]])
    
def get_last_layer_outputs(model: Model):
    return K.function([model.layers[0].input],
                      [model.layers[-2]])

def block(dimension: int, inp, filters: int, down_sample: bool = False):
    if dimension == 1:
        Conv = layers.Conv1D
    elif dimension == 2:
        Conv = layers.Conv2D
    elif dimension == 3:
        Conv = layers.Conv3D

    shorcut = inp
    strides = [2, 1] if down_sample else [1, 1]
    x = Conv(filters=filters, kernel_size=(3, 3), strides=strides[0], padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = Conv(filters=filters, kernel_size=(3, 3), strides=strides[1], padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    if down_sample:
        shorcut = Conv(filters=filters, kernel_size=(3, 3), strides=2, padding="same")(shorcut)
        shorcut = layers.BatchNormalization()(shorcut)
    
    x = layers.Add()([x, shorcut])
    x = layers.Activation("relu")(x)
    return x


def ResNet_like_model(
        input_shape: tuple, 
        structures: List[int], 
        name: str, 
        dimension: int = 1, 
        only_features_map: bool = False, 
        compile: bool = False, 
        show_size: bool = False) -> Model:
    if dimension == 1:
        Conv = layers.Conv1D
    elif dimension == 2:
        Conv = layers.Conv2D
    elif dimension == 3:
        Conv = layers.Conv3D

    if dimension == 1:
        GPool = layers.GlobalAvgPool1D
    elif dimension == 2:
        GPool = layers.GlobalAvgPool2D
    elif dimension == 3:
        GPool = layers.GlobalAvgPool3D
    
    inp = layers.Input(shape=input_shape)
    x = Conv(structures[0], kernel_size=3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    for idx, filters in enumerate(structures):
        down_sample = False
        if idx > 0 and filters[idx-1] < filters[idx]:
            down_sample = True
        x = block(
            dimension = dimension, 
            inp = x, 
            filters = filters, 
            down_sample = down_sample
        )

    x = GPool()(x)
    x = layers.Flatten()(x)
    if not only_features_map:
        x = layers.Dense(units=1, activation="sigmoid")(x)

    model = Model(
        inputs = inp,
        outputs = x,
        name = name,
    )
    
    if show_size:
        show_params(model, name)

    if compile:
        model.compile(
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["binary_accuracy"],
        )
    
    return model

def CNN_model(
        input_shape: tuple, 
        structures: List[Tuple[int, int]], 
        name: str, 
        dimension: int = 1, 
        only_features_map: bool = False, 
        compile: bool = False, 
        show_size: bool = False) -> Model:
    if dimension == 1:
        Conv = layers.Conv1D
    elif dimension == 2:
        Conv = layers.Conv2D
    elif dimension == 3:
        Conv = layers.Conv3D

    if dimension == 1:
        Pool = layers.MaxPool1D
    elif dimension == 2:
        Pool = layers.MaxPool2D
    elif dimension == 3:
        Pool = layers.MaxPool3D

    if dimension == 1:
        GPool = layers.GlobalMaxPool1D
    elif dimension == 2:
        GPool = layers.GlobalMaxPool2D
    elif dimension == 3:
        GPool = layers.GlobalMaxPool3D

    inp = layers.Input(shape=input_shape)
    x = Conv(
        filters = structures[0][0], 
        kernel_size = structures[0][1], 
        padding = "same", 
        activation = "relu",
        kernel_regularizer = reg.L2())(inp)
    x = layers.BatchNormalization()(x)
    x = Pool(pool_size=2)(x)
    x = layers.Dropout(rate=0.2)(x)
    
    
    for filters, kernel_size in structures[1::]:
        x = Conv(
            filters = filters, 
            kernel_size = kernel_size, 
            padding = "same", 
            activation = "relu",
            kernel_regularizer = reg.L2())(x)
        x = layers.BatchNormalization()(x)
        x = Pool(pool_size=2)(x)
        x = layers.Dropout(rate=0.2)(x)

    x = GPool()(x)
    x = layers.Flatten()(x)
    if not only_features_map:
        x = layers.Dense(1, activation="sigmoid")(x)

    model = Model(
        inputs = inp,
        outputs = x,
        name = name
    )
    
    if compile:
        model.compile(
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = ["binary_accuracy"],
        )
    
    
    if show_size:
        show_params(model, name)

    return model