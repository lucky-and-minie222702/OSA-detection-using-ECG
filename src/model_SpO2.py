# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import numpy as np
# import keras
# import sys
# import tensorflow as tf
# import pandas as pd
# from keras import Sequential, Model
# from keras import layers
# from os import path
# from keras.saving import load_model
# import argparse
# from keras.utils import to_categorical
# from keras import optimizers
# from sklearn.utils import shuffle
# from collections import Counter
# from keras import metrics
# from sklearn.model_selection import KFold
# import sklearn.preprocessing as prep
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from scikeras.wrappers import KerasClassifier
# import sklearn.model_selection as mdselect
# import keras.applications as apl
# import keras.regularizers as reg
# import joblib
# import tensorflow.python.keras.backend as K
# from sklearn.metrics import classification_report
# from data_functions import *
# from sklearn.metrics import confusion_matrix
# import datetime
# from model_functions import *
# from data_functions import *

# def reset_model(model):
#     weights = []
#     initializers = []
#     for layer in model.layers:
#         if isinstance(layer, (keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D, keras.layers.Conv3D)):
#             weights += [layer.kernel, layer.bias]
#             initializers += [layer.kernel_initializer, layer.bias_initializer]
#         elif isinstance(layer, keras.layers.BatchNormalization):
#             weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
#             initializers += [layer.gamma_initializer, layer.beta_initializer, layer.moving_mean_initializer, layer.moving_variance_initializer]
#         for w, init in zip(weights, initializers):
#             w.assign(init(w.shape, dtype=w.dtype))

# def block(inp, filters: int, down_sample: bool = False):
#     shorcut = inp
#     strides = [2, 1] if down_sample else [1, 1]
#     x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[0], padding="same")(inp)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides[1], padding="same")(x)
#     x = layers.BatchNormalization()(x)
    
#     if down_sample:
#         shorcut = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=2, padding="same")(shorcut)
#         shorcut = layers.BatchNormalization()(shorcut)
    
#     x = layers.Add()([x, shorcut])
#     x = layers.Activation("relu")(x)
#     return x

# def create_model():
#     inp = layers.Input(shape=(None, 1))
#     x = layers.Conv1D(32, kernel_size=3, activation="relu")(inp)
#     x = layers.MaxPool1D(pool_size=2)(x)
#     x = layers.Conv1D(64, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool1D(pool_size=2)(x)
#     x = layers.Conv1D(128, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool1D(pool_size=2)(x)
#     x = layers.Conv1D(256, kernel_size=3, activation="relu")(x)
#     x = layers.MaxPool1D(pool_size=2)(x)
#     x = layers.Conv1D(512, kernel_size=3, activation="relu")(x)
#     x = layers.GlobalMaxPool1D()(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(1, activation="sigmoid")(x)
    
#     model = Model(
#         inputs = inp,
#         outputs = x,
#         name="SpO2"
#     )
    
#     if "show_size" in sys.argv:
#         show_params(model, "SpO2")

#     model.compile(
#         optimizer = "adam",
#         loss = "binary_crossentropy",
#         metrics = ["binary_accuracy"],
#     )
    
#     return model


# save_path = path.join("res", "model_SpO2.keras")
# if "epochs" in sys.argv:
#     epochs = int(sys.argv[sys.argv.index("epochs")+1])
# else:
#     epochs = int(input("Please provide a valid number of epochs: "))
# batch_size = 16

# print("Creating model architecture...")
# model = create_model()

# print("Loading data...")
# X = np.vstack([np.load(path.join("gen_data", "SpO2_normal.npy")),
#                np.load(path.join("gen_data", "SpO2_apnea.npy"))])
# y = np.array([[0] * (len(X) // 2) +
#               [1] * (len(X) // 2)]).flatten()

# counts = Counter(y)
# print("Done!")
# print(f"\nTotal: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")

# indices = np.arange(len(y))
# np.random.shuffle(indices)

# X = X[indices]
# y = y[indices]

# if "num_cases" in sys.argv:
#     num_cases = sys.argv[sys.argv.index("num_cases")+1]
#     if num_cases != "all":
#         num_cases = int(num_cases)
#         X = X[:num_cases:]
#         y = y[:num_cases:]
# else:
#     num_cases = int(input("Please provide a valid number of cases for model to learn: "))
# if num_cases != "all":
#     indices = np.arange(num_cases)

# print(f"=> Training on {'full dataset' if num_cases == "all" else num_cases}")
# print(f"=> Training with {epochs} epochs")

# prompt = input("Continue? [y/N]: ")
# if prompt != "y":
#     exit()

# if sys.argv[1] == "std":
#     if "build" in sys.argv:
#         if not "id" in sys.argv:
#             id = input("Please provide an id for this section: ")
#         else:
#             id = sys.argv[sys.argv.index("id")+1]
    
#     print()
#     _s = f"| SECTION {id} |"
#     _space = " " * 3
#     print(_space + "=" * len(_s), _space + _s, _space + "=" * len(_s), sep="\n")
#     now = datetime.datetime.now()
#     print("Start at:", now, "\n")
#     id = str(now) + "_" + id
    
#     val_split = 0.1
#     train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=22022009)
    
#     y_train = y[train_indices]
#     X_train = X[train_indices]
    
#     y_test = y[test_indices]
#     X_test = X[test_indices]
    
#     count_train = Counter(y_train)
#     count_test = Counter(y_test)
#     print(f"=> Train set: Apnea cases [1]: {count_train[1]} - Normal cases [0]: {count_train[0]}")
#     print(f"=> Test set: Apnea cases [1]: {count_test[1]} - Normal cases [0]: {count_test[0]}")
#     print(f"=> Validation set: Apnea cases [1]: {int(count_train[1]*val_split)} - Normal cases [0]: {int(count_train[0]*val_split)}")

#     if "build" in sys.argv:
#         hist = model.fit(X_train, 
#                          y_train, 
#                          epochs = epochs, 
#                          batch_size = batch_size*2, 
#                          validation_split = val_split, 
#                          callbacks = [cb])
#         t = np.array(cb.logs)
#         np.save(path.join("history", f"{id}_train_time"), t)
#     elif "test" in sys.argv:
#         model = load_model(save_path)
#     print("Evaluating...")
#     pred = model.predict(X_test)
#     pred = [np.round(np.squeeze(x)) for x in pred]
#     f = open(path.join("history", f"{id}_result.txt"), "w")
#     print(classification_report(y_test, pred, target_names=["NO OSA", "OSA"]), file=f)
#     cm = list(confusion_matrix(y_test, pred))
#     print("Confusion matrix:", cm, file=f)
#     print("Metrics names:", model.metrics_names, file=f)
#     print("Loss and metrics ", model.evaluate(X_test, y_test, verbose=False), file=f)
#     f.close()
    
#     if "build" in sys.argv:
#         prompt = input("Enter \"save\" to save or anything else to discard: ")
#         if prompt == "save":
#             model.save(save_path)
#             for key, value in hist.history.items():
#                 data = np.array(value)
#                 save_path = path.join("history", f"{id}_SpO2_generate_{key}")
#                 np.save(save_path, data)
#             print("Saving done!")
#         else:
#             print("Discard!")

# elif sys.argv[1] == "full_build":
#     print("Training...")
#     hist = model.fit(X, y, epochs=epochs, batch_size=(batch_size * 2), validation_split=0.2)
#     print("Exporting...")
#     model.save(save_path)
#     for key, value in hist.history.items():
#         data = np.array(value)
#         save_path = path.join("history", f"SpO2_generate_{key}")
#         np.save(save_path, data)
#     print("Done!")
