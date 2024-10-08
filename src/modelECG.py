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
from librosa.feature import mfcc, delta

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

save_path = path.join("res", "modelECG.keras")
epochs = 5
batch_size = 16

model = create_model()

scaler = prep.MinMaxScaler()
def get_patients(plist):
    def get_patient(patientid):
        rec = np.load(path.join("numpy", f"patient_{patientid}.npy"))
        ann = np.load(path.join("numpy", f"annotation_{patientid}.npy"))
        return rec, ann

    X, y = get_patient(plist[0])
    siglen = len(y)
    plist = plist[1::]
    for i in plist:
        rec, ann = get_patient(i)
        X = np.hstack((X, rec))
        y = np.hstack((y, ann))
        siglen += len(ann)

    X = np.array(np.split(X, siglen))
    temp = []
    for x in X:
        mfccs = mfcc(y=x, sr=100, n_mfcc=24)
        delta1 = delta(mfccs, order=1)
        delta2 = delta(mfccs, order=2)
        data = np.concatenate([mfccs, delta1, delta2])
        temp.append(scaler.fit_transform(data))
    X = np.array(temp)
    X = np.expand_dims(X, 3)
    return X, y


kf = KFold(n_splits=5)
_X_total, _y_total = get_patients(range(1, 36))
counts = Counter(_y_total)
ideal = min(counts[0], counts[1])
c = [0 ,0] 
X_total, y_total = [], []
for X, y in zip(_X_total, _y_total):
    if c[y] <    ideal:
        c[y] += 1
        X_total.append(X)
        y_total.append(y)
X_total = np.array(X_total)
y_total = np.array(y_total)
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
    model.fit(X_total, y_total, epochs=epochs, batch_size=batch_size, verbose=False)
    print("Exporting...")
    model.save(save_path)
    print("Done!")
