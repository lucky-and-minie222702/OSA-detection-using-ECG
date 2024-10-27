import numpy as np
import sklearn.preprocessing as prep
from os import path
from librosa.feature import *
from scipy.signal import lfilter, savgol_filter

def feature_extract(X):
    scaler = prep.MinMaxScaler()
    temp = []
    for x in X:
        # x_f = savgol_filter(x, 2, 0)
        key = mfcc(y=x, hop_length=256, sr=100, n_mfcc=12)
        delta1 = delta(key, order=1)
        delta2 = delta(delta1, order=2)
        data = np.concatenate([key, delta1])
        # data = key
        temp.append(scaler.fit_transform(data))
    temp = np.array(temp)
    temp = np.expand_dims(temp, 3)
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