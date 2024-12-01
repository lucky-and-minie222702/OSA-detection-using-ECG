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
from timeit import default_timer as timer
import datetime

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

def show_params(model, name):
    print(f"Model {name}:")
    params = model.count_params()
    print(" | Total params:", "{:,}".format(params).replace(",", " "))
    print(" | Size        :", convert_bytes(params * 4))

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

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
cb = TimingCallback()

def create_model_ECG():
    # 2d + mfcc + tempogram
    inp = layers.Input(shape=(24, None, 4)) # the seccond dimension is the sampling rate
    
    x = layers.Conv2D(64, kernel_size=3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = block(x, 64)
    x = block(x, 64)
    x = block(x, 64)
    x = block(x, 128, True)
    x = block(x, 128)
    x = block(x, 128)
    x = block(x, 256, True)
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 512, True)
    x = block(x, 512)
    x = block(x, 512)
    
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(850, activation='relu')(x)

    model = Model(
        inputs = inp,
        outputs = x,
        name="ECG"
    )
    
    if "show_size" in sys.argv:
        show_params(model, "ECG")

    
    # model.compile(
    #     optimizer = "adam",
    #     loss = "binary_crossentropy",
    #     metrics = ["binary_accuracy"],
    # )
    
    return model

def create_model_SpO2():
    inp = layers.Input(shape=(16, ))
    
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    
    model = Model(
        inputs = inp,
        outputs = x,
        name="SpO2"
    )
    
    # model.compile(
    #     optimizer = "adam",
    #     loss = "binary_crossentropy",
    #     metrics = ["binary_accuracy"],
    # )
    
    if "show_size" in sys.argv:
        show_params(model, "SpO2")

    
    return model
    
def create_model():
    model_ECG = create_model_ECG()
    model_SpO2 = create_model_SpO2()
    merge = layers.concatenate([model_ECG.output, model_SpO2.output])
    merge = layers.Flatten()(merge)
    merge = layers.Reshape((30, 30, 1))(merge)
    
    merge = block(merge, 64)
    merge = block(merge, 64)
    merge = block(merge, 128, True)
    merge = block(merge, 128)
    merge = block(merge, 256, True)
    merge = block(merge, 256)

    merge = layers.AveragePooling2D(pool_size=2)(merge)

    merge = layers.Flatten()(merge)
    merge = layers.Dense(1, activation='sigmoid')(merge)
    
    model = Model(
        inputs = [model_ECG.input, model_SpO2.input],
        outputs = merge,
        name="ECG_SpO2"
    )
    
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["binary_accuracy", 
                   metrics.Precision(), 
                   metrics.Recall(),
                   metrics.TrueNegatives(),
                   metrics.TruePositives(),
                   metrics.FalseNegatives(),
                   metrics.FalsePositives()] ,
    )
    
    if "summary" in sys.argv:
        model.summary()
        
    if "show_size" in sys.argv:
        show_params(model, "Combined")
    
    return model

save_path = path.join("res", "model.keras")
epochs = 150 if "enhanced_training" in sys.argv else 100
batch_size = 16

print("Creating model architecture...")
model = create_model()

kf = KFold(n_splits=16)
print("Loading data...")
X_ECG = np.vstack([np.load(path.join("gen_data", "f_ECG_normal.npy")),
                     np.load(path.join("gen_data", "f_ECG_apnea.npy"))
                     ])
X_SpO2 = np.vstack([np.load(path.join("gen_data", "gs_SpO2_normal.npy")),
                     np.load(path.join("gen_data", "gs_SpO2_apnea.npy"))
                     ])
y_total = np.array([[0] * (len(X_ECG) // 2) +
                    [1] * (len(X_ECG) // 2)
                    ]).flatten()

counts = Counter(y_total)
print("Done!")
print(f"\nTotal: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")

indices = np.arange(len(y_total))
np.random.shuffle(indices)

X_ECG = X_ECG[indices]
X_SpO2 = X_SpO2[indices]
y_total = y_total[indices]

if "small_test" in sys.argv:
    print("=> Small testing mode!")
    small_size = 300
    epochs = 3
    X_ECG = X_ECG[:small_size:]
    X_SpO2 = X_SpO2[:small_size:]
    y_total = y_total[:small_size:]
    indices = np.arange(small_size)
else:
    print("=> Training on full dataset!")

if sys.argv[1] == "std":
    id = None
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
    print("Start at:", now)
    print("Estimated time:", convert_minutes(7.5*epochs), "\n")
    id = str(now) + "_" + id
    
    val_split = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=22022009)
    
    y_train = y_total[train_indices]
    X_ECG_train = X_ECG[train_indices]
    X_SpO2_train = X_SpO2[train_indices]
    
    y_test = y_total[test_indices]
    X_ECG_test = X_ECG[test_indices]
    X_SpO2_test = X_SpO2[test_indices]
    
    count_train = Counter(y_train)
    count_test = Counter(y_test)
    print(f"=> Train set: Apnea cases [1]: {count_train[1]} - Normal cases [0]: {count_train[0]}")
    print(f"=> Test set: Apnea cases [1]: {count_test[1]} - Normal cases [0]: {count_test[0]}")
    print(f"=> Validation set: Apnea cases [1]: {int(count_train[1]*val_split)} - Normal cases [0]: {int(count_train[0]*val_split)}")

    if "build" in sys.argv:
        hist = model.fit([X_ECG_train, X_SpO2_train], 
                         y_train, 
                         epochs=epochs, 
                         batch_size=batch_size*2, 
                         validation_split=val_split, 
                         callbacks=[cb])
        t = np.array(cb.logs)
        np.save(path.join("history", f"{id}_train_time"), t)
    elif "test" in sys.argv:
        model = load_model(save_path)
    print("Evaluating...")
    pred = model.predict([X_ECG_test, X_SpO2_test])
    pred = [np.round(np.squeeze(x)) for x in pred]
    f = open(path.join("history", f"{id}_result.txt"), "w")
    print(classification_report(y_test, pred, target_names=["NO OSA", "OSA"]), file=f)
    cm = list(confusion_matrix(y_test, pred))
    print("Confusion matrix:", cm, file=f)
    print("Metrics names:", model.metrics_names, file=f)
    print("Loss and metrics ", model.evaluate([X_ECG_test, X_SpO2_test], y_test, verbose=False), file=f)
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

elif sys.argv[1] == "full_build":
    print("Training...")
    hist = model.fit([X_ECG, X_SpO2], y_total, epochs=epochs, batch_size=(batch_size * 2), validation_split=0.2)
    print("Exporting...")
    model.save(save_path)
    for key, value in hist.history.items():
        data = np.array(value)
        save_path = path.join("history", f"SpO2_generate_{key}")
        np.save(save_path, data)
    print("Done!")
