import numpy as np
import sklearn.preprocessing as prep
from os import path
from librosa.feature import *
from scipy.signal import lfilter, savgol_filter
import sys

def feature_extract(X, verbose=False, contains_tempogram=False):
    scaler = prep.MinMaxScaler()
    temp = []
    hl = 256
    sr = 100
    total = len(X)
    count = 0
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
        if contains_tempogram:
            tempograms = tempogram(y=x, hop_length=hl, sr=sr, win_length=24)
        # final data
        data = np.stack([
            mfccs1, 
            mfccs2, 
            mfccs3,
        ] + [tempograms] if contains_tempogram else [], axis=2)
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

def get_patients_SpO2(plist):
    def get_patient(patientid):
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

def get_patients_ECG(plist):
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
    return X, y