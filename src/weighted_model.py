import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mne import verbose
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
from data_functions import *
import time

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


kf = KFold(n_splits=8)
save_path = path.join("res", "model_weighted.keras")
epochs = 5
batch_size = 32

model_ECG = load_model(path.join("res", "model_ECG.keras"))
model_SpO2 = load_model(path.join("res", "model_SpO2.keras"))

reset_model(model_ECG)
reset_model(model_SpO2)

X_pair = np.load(path.join("gen_data", "rec_pair_data.npy"))
X_pair = np.split(X_pair, 2, axis=2)

X_ECG = X_pair[0].squeeze()
# X_ECG = feature_extract(X_ECG, verbose=False, contains_tempogram=True)
X_SpO2 = X_pair[1].squeeze()
X_pair = []

model_pre_fitted = load_model(path.join("res", "model_ECG.keras"))
reset_model(model_pre_fitted)

y_total = np.load(path.join("gen_data", "ann_pair_data.npy"))
m = keras.metrics.BinaryAccuracy()

if "pre_fit" in sys.argv:
    X_prefit = np.load(path.join("gen_data", "rec_remain_ECG.npy"))
    y_prefit = np.load(path.join("gen_data", "ann_remain_ECG.npy"))
    model_pre_fitted.fit(
        X_prefit, y_prefit,
        batch_size=batch_size,
        epochs=epochs,
    )
    X_prefit = []
    model_pre_fitted.save(path.join("res", "prefitted_model_ECG.keras"))
    print("Model ECG has been pre-fitted and save!")
else:
    model_pre_fitted = load_model(path.join("res", "prefitted_model_ECG.keras"))

if "test" in sys.argv:
    for i, (train_index, test_index) in enumerate(kf.split(X_ECG)):
        if i > 3:
            break
        X_e = X_ECG[train_index]
        X_s = X_SpO2[train_index]
        y = y_total[train_index]
        counts = Counter(y)
        print(f"Fold {i+1}:")
        print(f"=> Train set: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")
        y = np.expand_dims(y, 1)

        model_ECG.fit(X_e, y, epochs=epochs, batch_size=batch_size, verbose=False)
        model_SpO2.fit(X_s, y, epochs=epochs, batch_size=batch_size, verbose=False)

        X_e = X_ECG[test_index]
        X_s = X_SpO2[test_index]
        y = y_total[test_index]

        counts = Counter(y)
        print(f"=> Test set: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")
        pred_e = model_ECG.predict(X_e, verbose=False).flatten()
        pred_s = model_SpO2.predict(X_s, verbose=False).flatten()

        for rate in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            pred = pred_e * rate + pred_s * round(1 - rate, 1)
            m.update_state(y, pred)
            score = m.result()
            print(f"W_ECG = {rate} - W_SpO2 = {round(1 - rate, 1)} => Accuracy (correct / total): {score}")

        model_ECG.set_weights(model_pre_fitted.get_weights())
        reset_model(model_SpO2)
