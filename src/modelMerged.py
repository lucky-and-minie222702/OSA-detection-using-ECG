import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.saving import load_model
from os import path
import sklearn.preprocessing as prep
from librosa.feature import mfcc, delta

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
    
    X = np.array(np.split(X, siglen, axis=0))
    X = np.array([rec.T for rec in X])
    return X, y

def get_patients_ECG(plist):
    scaler = prep.MinMaxScaler()
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
    temp = []
    for x in X:
        
        mfccs = mfcc(y=x, sr=100, n_mfcc=24)
        delta1 = delta(mfccs, order=1)
        delta2 = delta(mfccs, order=2)
        data = np.concatenate([mfccs, delta1, delta2])
        temp.append(scaler.fit_transform(data))
    X = np.array(temp)
    X = np.expand_dims(X, 3)
    return X, y

def calc_pred(pred1, pred2, weight1, weight2):
    pred1 = np.squeeze(pred1)
    pred2 = np.squeeze(pred2)
    res = pred1 * weight1 + pred2 * weight2
    return (res)

# test data
X_ECG, y_ECG = get_patients_ECG([1])
X_SpO2, y_SpO2 = get_patients_SpO2([1])

ECG_path = path.join("res", "modelECG.keras")
SpO2_path = path.join("res", "modelSpO2.keras")

ECG_model = load_model(ECG_path)
SpO2_model = load_model(SpO2_path)

# print(X_ECG.shape, X_SpO2.shape)
pred1 = ECG_model.predict(X_ECG)[20]
pred2 = SpO2_model.predict(X_SpO2)[0]
print(pred1, pred2)
pred = calc_pred(pred1, pred2, 0.8, 0.2)
print(pred)