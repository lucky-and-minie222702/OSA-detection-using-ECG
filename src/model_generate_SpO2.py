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
from data_functions import *

def reset_model(model):
    weights = []
    initializers = []
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv1D, keras.layers.Conv3D)):
            weights += [layer.kernel, layer.bias]
            initializers += [layer.kernel_initializer, layer.bias_initializer]
        elif isinstance(layer, keras.layers.BatchNormalization):
            weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
            initializers += [layer.gamma_initializer, layer.beta_initializer, layer.moving_mean_initializer, layer.moving_variance_initializer]
        for w, init in zip(weights, initializers):
            w.assign(init(w.shape, dtype=w.dtype))

def create_model():
    inp = layers.Input(shape=(None, 1))
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu")(inp)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=512, kernel_size=3, activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(16)(x)
    
    model = Model(
        inputs = inp,
        outputs = x,
        name = "generate_SpO2"
    )
    
    model.compile(
        optimizer = "adam",
        loss = keras.losses.Huber(),
        metrics = ["mse", "mae"],
    )
    
    model.summary()
    return model

save_path = path.join("res", "model_generate_SpO2.keras")
epochs = 15
batch_size = 32

model = create_model()

print("Loading data...")
X = np.load(path.join("gen_data", "ECG-pair.npy"))
y = np.load(path.join("gen_data", "y-pair.npy"))
print("Done!")

if "build" in sys.argv:
    hist = model.fit(
        X, y,
        batch_size = batch_size,
        epochs = epochs,
    )
    print("Exporting...")
    model.save(save_path)
    for key, value in hist.history.items():
        data = np.array(value)
        save_path = path.join("history", f"SpO2_generate_{key}")
        np.save(save_path, data)
    print("Done!")



#clf = KerasClassifier(create_model(), 
#    epochs=epochs,
#    batch_size=batch_size,
#    verbose=False,
#)