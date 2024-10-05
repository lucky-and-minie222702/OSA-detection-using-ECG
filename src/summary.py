import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.saving import load_model
from keras.utils import plot_model
from os import path


ECG_path = path.join("res", "modelECG.keras")
SpO2_path = path.join("res", "modelSpO2.keras")

sECG_path = path.join("summary", "modelECG.png")
sSpO2_path = path.join("summary", "modelSpO2.png")

ECG_model = load_model(ECG_path)
SpO2_model = load_model(SpO2_path)

plot_model(ECG_model, sECG_path)
plot_model(SpO2_model, sSpO2_path)
