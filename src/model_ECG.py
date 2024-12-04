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
from sklearn.metrics import confusion_matrix
import datetime
from model_functions import *
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

def create_model_raw():
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (32, 13),
            (64, 11),
            (128, 7),
            (256, 5),
            (512, 3)
        ],
        name = "ECG_raw",
        dimension = 1,
        show_size = True
    )

def create_model_fft() -> Model:
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (32, 7),
            (64, 5),
            (128, 5),
            (256, 3),
            (512, 3)
        ],
        name = "ECG_fft",
        dimension = 1,
        show_size = True
    )

def create_model_psd() -> Model:
    return CNN_model(
        input_shape = (None, 2),
        structures = [
            (32, 3),
            (64, 3),
            (128, 3),
            (256, 5),
            (512, 3)
        ],
        name = "ECG_psd",
        dimension = 1,
        show_size = True
    )

def create_model():
    raw_model = create_model_raw()
    fft_model = create_model_fft()
    psd_model = create_model_psd()
    
    out = layers.concatenate([
        raw_model.output,
        fft_model.output,
        psd_model.output,
    ])
    out = layers.Dense(1024, activation="relu")(out)
    out = layers.Dense(1, activation="relu")(out)
    
    model = Model(
        inputs = [raw_model.input, fft_model.input, psd_model.input],
        outputs = out,
        name="ECG_combined"
    )
    
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = [
            "binary_accuracy",
            metrics.TruePositives(),
            metrics.TrueNegatives(),
            metrics.FalsePositives(),
            metrics.FalseNegatives(),
            metrics.AUC(),
        ],
    )

    if "show_size" in sys.argv:
        show_params(model, "ECG_combined")

    return model

save_path = path.join("res", "model_ECG.keras")
if "epochs" in sys.argv:
    epochs = int(sys.argv[sys.argv.index("epochs")+1])
else:
    epochs = int(input("Please provide a valid number of epochs: "))
batch_size = 64

print("Creating model architecture...")
model = create_model()

print("Loading data...")

is_data_augmented = "augmented" in sys.argv
X_raw = np.vstack([np.load(path.join("gen_data", f"{'a_' if is_data_augmented else ''}ECG_normal.npy")), np.load(path.join("gen_data", f"{'a_' if is_data_augmented else ''}ECG_apnea.npy"))])
X_fft = np.vstack([np.load(path.join("gen_data", "fft_ECG_normal.npy")), np.load(path.join("gen_data", "fft_ECG_apnea.npy"))])
X_psd = np.vstack([np.load(path.join("gen_data", "psd_ECG_normal.npy")), np.load(path.join("gen_data", "psd_ECG_apnea.npy"))])

y = np.array([[0] * (len(X_raw) // 2) + [1] * (len(X_raw) // 2)]).flatten()

counts = Counter(y)
print("Done!")
print(f"Total: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")

indices = np.arange(len(y))
np.random.shuffle(indices)

X_raw = X_raw[indices]
X_fft = X_fft[indices]
X_psd = X_psd[indices]

if "num_cases" in sys.argv:
    num_cases = sys.argv[sys.argv.index("num_cases")+1]
    if num_cases != "all":
        num_cases = int(num_cases)
        X_raw = X_raw[:num_cases:]
        X_fft = X_fft[:num_cases:]
        X_psd = X_psd[:num_cases:]
        y = y[:num_cases:]
else:
    num_cases = int(input("Please provide a valid number of cases for model to learn: "))
if num_cases != "all":
    indices = np.arange(num_cases)

print(f"=> Training on {'full dataset' if num_cases == 'all' else num_cases}")
print(f"=> Training with {epochs} epochs")

prompt = input("Continue? [y/N]: ")
if prompt != "y":
    exit()

if sys.argv[1] == "std":
    if "build" in sys.argv:
        if not "id" in sys.argv:
            id = input("Please provide an id for this section: ")
        else:
            id = sys.argv[sys.argv.index("id")+1]
    
    print()
    _s = f"| SECTION {id} |"
    _space = " " * 3
    print(_space + "=" * len(_s), _space + _s, _space + "=" * len(_s), sep="\n")
    now = datetime.datetime.now()
    print("Start at:", now, "\n")
    id = str(now) + "_" + id
    
    val_split = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=22022009)
    
    y_train = y[train_indices]
    X_raw_train = X_raw[train_indices]
    X_fft_train = X_fft[train_indices]
    X_psd_train = X_psd[train_indices]
    
    y_test = y[test_indices]
    X_raw_test = X_raw[test_indices]
    X_fft_test = X_fft[test_indices]
    X_psd_test = X_psd[test_indices]
    
    count_train = Counter(y_train)
    count_test = Counter(y_test)
    print(f"=> Train set: Apnea cases [1]: {count_train[1]} - Normal cases [0]: {count_train[0]}")
    print(f"=> Test set: Apnea cases [1]: {count_test[1]} - Normal cases [0]: {count_test[0]}")
    print(f"=> Validation set: Apnea cases [1]: {int(count_train[1]*val_split)} - Normal cases [0]: {int(count_train[0]*val_split)}")

    if "build" in sys.argv:
        hist = model.fit([X_raw_train, X_fft_train, X_psd_train], 
                         y_train, 
                         epochs = epochs, 
                         batch_size = batch_size, 
                         validation_split = val_split, 
                         callbacks = [cb])
        t = sum(cb.logs)
        print(f"Total training time: {t} seconds")
    elif "test" in sys.argv:
        model = load_model(save_path)
    print("Evaluating...")
    pred = model.predict([X_raw_test, X_fft_test, X_psd_test])
    pred = [np.round(np.squeeze(x)) for x in pred]
    f = open(path.join("history", f"{id}_result.txt"), "w")
    print(classification_report(y_test, pred, target_names=["NO OSA", "OSA"]), file=f)
    cm = list(confusion_matrix(y_test, pred))
    print("Confusion matrix:", cm, file=f)
    print("Metrics names:", model.metrics_names, file=f)
    print("Loss and metrics ", model.evaluate([X_raw_test, X_fft_test, X_psd_test], y_test, verbose=False), file=f)
    f.close()
    
    if "build" in sys.argv:
        prompt = input("Enter \"save\" to save or anything else to discard: ")
        if prompt == "save":
            model.save(save_path)
            for key, value in hist.history.items():
                data = np.array(value)
                save_path = path.join("history", f"{id}_SpO2_generate_{key}")
                np.save(save_path, data)
            print("Saving done!")
        else:
            print("Discard!")