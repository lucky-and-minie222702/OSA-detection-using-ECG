import numpy as np
import sklearn.preprocessing as prep
from os import path
from librosa.feature import mfcc, delta

def to_mfcc(X):
    scaler = prep.MinMaxScaler()
    temp = []
    for x in X:
        mfccs = mfcc(y=x, sr=100, n_mfcc=24)
        delta1 = delta(mfccs, order=1)
        delta2 = delta(mfccs, order=2)
        data = np.concatenate([mfccs, delta1, delta2])
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