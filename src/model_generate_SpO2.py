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

def reset_model(model):
    weights = []
    initializers = []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D, keras.layers.Conv3D)):
            weights += [layer.kernel, layer.bias]
            initializers += [layer.kernel_initializer, layer.bias_initializer]
        elif isinstance(layer, keras.layers.BatchNormalization):
            weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
            initializers += [layer.gamma_initializer, layer.beta_initializer, layer.moving_mean_initializer, layer.moving_variance_initializer]
        for w, init in zip(weights, initializers):
            w.assign(init(w.shape, dtype=w.dtype))

def block(inp, filters, down_sample=False):
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

def create_model():
    # 2d + mfcc + tempogram
    inp = layers.Input(shape=(24, None, 4)) # the seccond size is sampling rate dimension
    
    x = layers.Conv2D(64, kernel_size=3)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = block(x, 64)
    x = block(x, 64)
    x = block(x, 128, True)
    x = block(x, 128)
    
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2, activation="relu")(x)

    model = Model(inputs=inp, outputs=x, name="NHCT")
    
    model.compile(
        optimizer="adam",
        loss="mean_absolute_error",
        metrics=["mean_squared_error"],
    )
    
    # model.summary()
    return model

save_path = path.join("res", "model_generate_SpO2.keras")
epochs = 5
batch_size = 32

X_total = np.vstack([np.load(path.join("gen_data", "f_ECG_normal.npy")),
                     np.load(path.join("gen_data", "f_ECG_apnea.npy"))
                     ])
y_total = np.vstack([np.load(path.join("gen_data", "SpO2_normal.npy")),
                     np.load(path.join("gen_data", "SpO2_apnea.npy"))
                     ])
y_total = np.array([
    [np.mean(y), np.std(y), np.max(y), np.min(y)] for y in y_total
])