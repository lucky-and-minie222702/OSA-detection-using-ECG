import signal
import numpy as np
import sklearn.preprocessing as prep
from os import path
from librosa.feature import *
from scipy.signal import lfilter, savgol_filter
import sys
import scipy.stats as stats
from scipy.signal import find_peaks, hilbert, welch
from scipy.fft import fft, ifft
import pywt
import joblib
from typing import Tuple, List

def show_process_bar(count: int, total: int):
    percent = round(count / total * 100, 1)
    loaded = "=" * (int(percent) //2)
    if loaded != "" and count < total:
        loaded = loaded[:-1:] + ">"
    unloaded = " " * (50 - int(percent) // 2)
    print(f" {str(percent):>4}% [{loaded}{unloaded}]", "Inputs:", count, "/", total, end="\r")
    sys.stdout.flush()

class Legend:
    def extract_features1(X: np.ndarray, sampling_rate: int =  100, contains_tempogram: bool = False, verbose:bool = False) -> np.ndarray:
        temp = []
        hl = 256
        sr = sampling_rate
        total = len(X)
        count = 0
        
        for x in X:
            # mfcc dct 1
            mfccs1 = mfcc(y=x, hop_length=hl, sr=sr, n_mfcc=12, dct_type=1)
            delta1 = delta(mfccs1, order=1)
            mfccs1 = np.concatenate([mfccs1, delta1])
            # mfcc dct 2
            mfccs2 = mfcc(y=x, hop_length=hl, sr=sr, n_mfcc=12, dct_type=2)
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
                show_process_bar(count, total)
        if verbose:
            print()
        
        temp = np.array(temp)
        return temp

    def extract_features2(X: np.ndarray, sampling_rate: int =  100, verbose:bool = False) -> np.ndarray:
        total = len(X)
        count = 0
        res_fft = []
        res_psd = []
        
        for signal in X:
            _fft = fft(signal).real.astype("float64")
            _fft = np.abs(_fft)[1::] # remove outliers
            _freqs, _psd = welch(signal, fs=sampling_rate)
            res_fft.append(_fft)
            res_psd.append(np.array([_freqs, _psd]).T)
            
            count += 1
            if verbose:
                show_process_bar(count, total)

        if verbose:
            print()
            
        return np.array(res_fft)
    
    def extract_stats(signals, sampling_rate: int = 100, save_scaler: bool = False, verbose: bool = False, name: str = "", use_scaler = None, using_frequency_components: bool = False):
        val = []
        keys = []
        count = 0
        total = len(signals)
        
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
            if using_frequency_components:
                freqs, psd = welch(signal, fs=sampling_rate)
                features['psd_mean'] = np.mean(psd)
                features['psd_max'] = np.max(psd)
                features['dominant_frequency'] = freqs[np.argmax(psd)]
                coeffs = pywt.wavedec(signal, 'haar', level=3)
                features['wavelet_energy'] = sum(np.sum(c**2) for c in coeffs)
            
            val.append(list(features.values()))
            keys = list(features.keys())

            # Progress
            count += 1
            if verbose:
                show_process_bar(count, total)
        if verbose:
            print()

        val = np.array(val)
        if use_scaler is None:
            scaler = prep.MinMaxScaler()
            val = scaler.fit_transform(val)
        else:
            scaler = joblib.load(path.join("res", f"{use_scaler}_stats_scaler.scaler"))
            val = scaler.transform(val)
        if save_scaler:
            joblib.dump(scaler, path.join("res", f"{name}_stats_scaler.scaler"))
        
        return val , keys

def get_patients_SpO2(plist: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    def get_patient(patientid: int) -> Tuple[np.ndarray, np.ndarray]:
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

def get_patients_ECG(plist: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    def get_patient(patientid: int) -> Tuple[np.ndarray, np.ndarray]:
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

