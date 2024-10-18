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
from data_functions import *

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

def create_model():
    inp = layers.Input(shape=(72, 12, 1))
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inp)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Reshape((1,) + x.shape[1:])(x)
    if "compare" in sys.argv:
        x = layers.Bidirectional(layers.LSTM(units=3, return_sequences=True))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x, name="NHCT")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    if sys.argv[1] != "report":
        model.summary()
    return model

save_path = path.join("res", "model_ECG.keras")
epochs = 5
batch_size = 16

model = create_model()

kf = KFold(n_splits=5)
X_total = np.vstack([np.load(path.join("gen_data", "ECG_normal.npy")),
                     np.load(path.join("gen_data", "ECG_apnea.npy"))
                     ])
y_total = np.array([[0] * (len(X_total) // 2) +
                    [1] * (len(X_total) // 2)
                    ]).flatten()
X_total = to_mfcc(X_total)
counts = Counter(y_total)
print(f"\nTotal: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")

X_total, y_total = shuffle(X_total, y_total, random_state=27022009)
scores = []
if sys.argv[1] == "test" or sys.argv[1] == "report":
    for i, (train_index, test_index) in enumerate(kf.split(X_total)):
        X = X_total[train_index]
        y = y_total[train_index]
        counts = Counter(y)
        print(f"Fold {i+1}:")
        print(f"=> Train set: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")
        y = np.expand_dims(y, 1)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=False)
        X = X_total[test_index]
        y = y_total[test_index]
        counts = Counter(y)
        print(f"=> Test set: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")
        if sys.argv[1] == "test":
            y = np.expand_dims(y, 1)
            score = np.round(model.evaluate(X, y, batch_size=batch_size, verbose=False)[1], 3)
            scores.append(score)
            print(f"Accuracy (correct / total): {score}\n{"-"*50}")
        else:
            pred = model.predict(X, verbose=False)
            pred = [np.round(np.squeeze(x)) for x in pred]
            print(classification_report(y, pred, target_names=["NO OSA", "OSA"]))
        # reset
        reset_model(model)
    if sys.argv[1] == "test":
        print("*** SUMMARY ***")
        for i, score in enumerate(scores):
            print(f"Fold {i+1}: Accuracy: {score}")
        print(f"Average accuracy: {np.round(np.mean(scores, axis=0), 3)}")

elif sys.argv[1] == "build":
    print("Training...")
    model.fit(X_total, y_total, epochs=epochs, batch_size=(batch_size * 2), validation_split=0.2)
    print("Exporting...")
    model.save(save_path)
    print("Done!")
