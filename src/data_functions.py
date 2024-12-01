import numpy as np
import sklearn.preprocessing as prep
from os import path
from librosa.feature import *
from scipy.signal import lfilter, savgol_filter
import sys
import scipy.stats as stats
from scipy.signal import find_peaks, hilbert, welch
import pywt
import joblib
from timeit import default_timer as timer
import keras

def extract_stats(signals, sampling_rate: int = 100, verbose: bool = False):
    val = []
    keys = []
    count = 0
    total = len(signals)
    print("Extracting statistics...")
    
    for signal in signals:
        features = {}
        signal = signal.flatten()
        features['mean'] = np.mean(signal)
        features['median'] = np.median(signal)
        features['std_dev'] = np.std(signal)
        features['variance'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['max'] = np.max(signal)
        features['min'] = np.min(signal)
        features['range'] = features['max'] - features['min']
        peaks, _ = find_peaks(signal)
        features['num_peaks'] = len(peaks)
        features['peak_mean'] = np.mean(signal[peaks]) if len(peaks) > 0 else 0
        zero_crossings = np.where(np.diff(np.sign(signal - np.mean(signal))) != 0)[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        freqs, psd = welch(signal, fs=sampling_rate)
        features['psd_mean'] = np.mean(psd)
        features['psd_max'] = np.max(psd)
        features['dominant_frequency'] = freqs[np.argmax(psd)]
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        features['envelope_mean'] = np.mean(amplitude_envelope)
        coeffs = pywt.wavedec(signal, 'haar', level=3)
        features['wavelet_energy'] = sum(np.sum(c**2) for c in coeffs)
        
        val.append(list(features.values()))
        keys = list(features.keys())

        # Progress
        count += 1
        if verbose:
            percent = int(count / total * 100)
            loaded = "=" * (percent//2)
            if loaded != "" and count < total:
                loaded = loaded[:-1:] + ">"
            unloaded = " " * (50 - (percent//2))
            print(f" {percent:3d}% [{loaded}{unloaded}]", "Inputs:", count, "/", total, end="\r")
            sys.stdout.flush()
    if verbose:
        print()

    val = np.array(val)
    
    return val , keys


def extract_features(X: np.ndarray, sampling_rate: int =  100, contains_tempogram: bool = False, verbose:bool = False) -> np.ndarray:
    temp = []
    hl = 256
    sr = sampling_rate
    total = len(X)
    count = 0
    print("Extracting features...")
    for x in X:
        # mfcc dct 1
        mfccs1 = mfcc(y=x, hop_length=hl, sr=sr, n_mfcc=12, dct_type=1)
        delta1 = delta(mfccs1, order=1)
        mfccs1 = np.concatenate([mfccs1, delta1])
        # mfcc dct 2
        mfccs2 = mfcc(y=x, hop_length=hl, sr=sr, n_mfcc=12, dct_type=1)
        delta1 = delta(mfccs2, order=1)
        mfccs2 = np.concatenate([mfccs2, delta1])
        # mfcc dct 3
        mfccs3 = mfcc(y=x, hop_length=hl, sr=sr, n_mfcc=12, dct_type=3)
        delta1 = delta(mfccs3, order=1)
        mfccs3 = np.concatenate([mfccs3, delta1])
        # tempogram
        tempograms = []
        if contains_tempogram:
            tempograms = [tempogram(y=x, hop_length=hl, sr=sr, win_length=24)]
        # final data
        data = np.stack([
            mfccs1, 
            mfccs2, 
            mfccs3,
        ] + tempograms, axis=2)
        temp.append(data)
        # Progress
        count += 1
        if verbose:
            percent = int(count / total * 100)
            loaded = "=" * (percent//2)
            if loaded != "" and count < total:
                loaded = loaded[:-1:] + ">"
            unloaded = " " * (50 - (percent//2))
            print(f" {percent:3d}% [{loaded}{unloaded}]", "Inputs:", count, "/", total, end="\r")
            sys.stdout.flush()
    if verbose:
        print()
    
    temp = np.array(temp)
    return temp

def get_patients_SpO2(plist: list) -> tuple:
    def get_patient(patientid: int) -> tuple:
        rec = np.load(path.join("numpy", f"SpO2_patient_{patientid}.npy"))
        ann = np.load(path.join("numpy", f"SpO2_annotation_{patientid}.npy"))
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
    return X, y

def get_patients_ECG(plist: list) -> tuple:
    def get_patient(patientid: int) -> tuple:
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
    return X, y

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()