import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.saving import load_model
from keras.utils import plot_model
from os import path

ECG_path = path.join("res", "modelECG.keras")
SpO2_path = path.join("res", "modelSpO2.keras")

ECG_model = load_model(ECG_path)
SpO2_model = load_model(SpO2_path)

sECG_path = path.join("summary", "modelECG.png")
sSpO2_path = path.join("summary", "modelSpO2.png")

plot_model(ECG_model, sECG_path, show_shapes=True, show_layer_activations=True)
plot_model(SpO2_model, sSpO2_path, show_shapes=True, show_layer_activations=True)
