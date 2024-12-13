import os
import sys
if "disable_XLA" in sys.argv:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
if "lazy_loading" in sys.argv:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
if "disable_GPU" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if "no_logs" in sys.argv:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
# from scikeras.wrappers import KerasClassifier
import sklearn.model_selection as mdselect
import keras.applications as apl
import keras.regularizers as reg
import joblib
import tensorflow.python.keras.backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
import keras
from typing import List, Tuple, Any
import keras.callbacks as cbk
import math

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
else:
    print("No GPU detected. Using CPU.")

class EpochProgressCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} completed!", end="\r")

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def convert_bytes(byte_size: int) -> str:
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = byte_size
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def convert_seconds(total_seconds: float) -> str:
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

def block(dimension: int, inp, filters: int, down_sample: bool = False, layers_activation = layers.Activation("relu")):
    if dimension == 1:
        Conv = layers.Conv1D
    elif dimension == 2:
        Conv = layers.Conv2D
    elif dimension == 3:
        Conv = layers.Conv3D

    shorcut = inp
    strides = [2, 1] if down_sample else [1, 1]
    x = Conv(filters, 3, strides[0], padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers_activation(x)
    x = Conv(filters, 3, strides[1], padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    if down_sample:
        shorcut = Conv(filters, 3, 2, padding="same")(shorcut)
        shorcut = layers.BatchNormalization()(shorcut)
    
    x = layers.Add()([x, shorcut])
    x = layers_activation(x)
    return x

def CNN_model(
        input_shape: tuple, 
        structures: List[Tuple[int, int, float]], 
        decoder_structures: List[Tuple[int, float]],
        name: str, 
        layers_activation = layers.Activation("relu"),
        dimension: int = 1, 
        features: int = 512,
        only_features_map: bool = False, 
        compile: bool = False,
        show_size: bool = False,
        use_batch_norm_in_Conv: bool = True,
        use_batch_norm_in_FC: bool = False,
        custom_input = None) -> Tuple[Model, Any, Any] :
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

    if custom_input is None:
        inp = layers.Input(shape=input_shape)
    else:
        inp = custom_input
    
    encoder = layers.Normalization()(inp)
    for filters, kernel_size, dropout_rate, pool_size in structures:
        encoder = Conv(
            filters = filters, 
            kernel_size = kernel_size, 
            padding = "same", 
            kernel_regularizer = reg.L2())(encoder)
        if use_batch_norm_in_Conv:
            encoder = layers.BatchNormalization()(encoder)
        encoder = layers_activation(encoder)
        encoder = Pool(pool_size=pool_size)(encoder)
        encoder = layers.Dropout(rate=dropout_rate)(encoder)
    encoder = GPool()(encoder)
    encoder = layers.Flatten()(encoder)
    encoder = layers.Dense(features, activation="tanh" if only_features_map else layers_activation)(encoder)
    
    decoder = layers.Dense(decoder_structures[0][0])(encoder)
    decoder = layers_activation(decoder)
    if use_batch_norm_in_FC:
        decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dropout(rate=decoder_structures[0][1])(decoder)
    for units, dropout_rate in decoder_structures[1::]:
        decoder = layers.Dense(units)(decoder)
        if use_batch_norm_in_FC:
            decoder = layers.BatchNormalization()(decoder)
        decoder = layers_activation(decoder)
        decoder = layers.Dropout(rate=dropout_rate)(decoder)
    decoder = layers.Dense(1, activation="sigmoid")(decoder)

    model = Model(
        inputs = inp,
        outputs = encoder if only_features_map else decoder,
        name = name
    )
    
    if compile:
        model.compile(
            optimizer = "adam",
            loss = "binary_crossentropy",
            metrics = [
                metrics.BinaryAccuracy(name = f"threshold_0.{t}",
                                       threshold = t/10) for t in range(1, 10)
            ],
        )
    
    if show_size:
        show_params(model, name)

    return (model, encoder, decoder)

def forget(model, forget_rate: float):
    weights = model.get_weights()
    for i in range(len(weights)):
        if len(weights[i].shape) > 1: # ignore biases
            mask = np.random.rand(*weights[i].shape) > forget_rate # Set rate% to zero randomly
            weights[i] = weights[i] * mask
    model.set_weights(weights)
    
class RandomForget(cbk.Callback):
    def __init__(self, layer_names = None, forget_rate: float = 0.2, remember_factor: float = 0.8):
        super().__init__()
        self.layer_names = layer_names
        self.forget_rate = forget_rate
        self.remember_factor = remember_factor

    def on_epoch_begin(self, epoch: int, logs = None):
        for layer in self.model.layers:
            if self.layer_names is None or layer.name in self.layer_names:
                if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                    for weight_tensor in layer.trainable_weights:
                        weights = weight_tensor.numpy()
                        mask = np.random.rand(*weights.shape) > self.forget_rate
                        updated_weights = weights * mask
                        weight_tensor.assign(updated_weights)
        self.forget_rate *= self.remember_factor
        
class DynamicWeightSparsification(cbk.Callback):
    def __init__(self, sparsity_target: float, end_from_epoch: int = None, layer_names = None, show_logs: bool = False):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.layer_names = layer_names
        self.end = end_from_epoch
        self.show_logs = show_logs

    def on_epoch_end(self, epoch: int, logs = None):
        if epoch >= self.end and not self.end is None:
            return
        for layer in self.model.layers:
            if self.layer_names is None or layer.name in self.layer_names:
                if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                    for weight_tensor in layer.trainable_weights:
                        weights = weight_tensor.numpy()
                        threshold = np.percentile(np.abs(weights), self.sparsity_target * 100)
                        mask = np.abs(weights) >= threshold
                        sparsified_weights = weights * mask
                        weight_tensor.assign(sparsified_weights)
                        if self.show_logs:
                            print(f"Sparsified {layer.name} weights: {np.sum(mask == 0)} zeros added.")
                        
class WeightMemoryMechanism(cbk.Callback):
    def __init__(self, patience: int = 3, monitor: str = 'val_loss', start_from_epoch: int = 0, show_logs: bool = False):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.best_weights = None
        self.best_epoch = -1
        self.best_score = np.inf if 'loss' in monitor else -np.inf
        self.wait = 0
        self.show_logs = show_logs
        self.start = start_from_epoch

    def on_epoch_end(self, epoch: int, logs = None):
        if epoch < self.start:
            return
        current_score = logs.get(self.monitor)
        if current_score is not None:
            if ('loss' in self.monitor and current_score < self.best_score) or('accuracy' in self.monitor and current_score > self.best_score):
                self.best_weights = self.model.get_weights()
                self.best_score = current_score
                self.best_epoch = epoch
                self.wait = 0
                if self.show_logs:
                    print(f"Saved best weights at epoch {epoch + 1}.")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    if self.show_logs:
                        print(f"Restoring best weights from epoch {self.best_epoch + 1}.")
                    self.model.set_weights(self.best_weights)
                    self.wait = 0