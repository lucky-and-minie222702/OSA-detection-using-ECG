import numpy as np
from os import path
import os
import sys
import argparse
import wfdb
from librosa.feature import mfcc, delta, poly_features, tempogram
import sklearn.preprocessing as prep

if not path.isdir("numpy"):
    os.makedirs("numpy")

print("*** Make sure that you have changed the dataset directory name to \"database\" before")

f = open(path.join("database", "list"))
records = f.read().splitlines()

total_sig_len = 0 
total_minutes = 0

scaler = prep.MinMaxScaler()
for i in range(len(records)):
    if records[i][0] == "x":
        break

    rec = wfdb.rdsamp(path.join("database", records[i]))
    ann = wfdb.rdann(path.join("database", records[i]), extension="apn").symbol
    
    ann = np.array([1 if x == "A" else 0 for x in ann])
    ann.resize(len(ann)-1)
    info = rec[1]
    siglen = len(ann) * 6000
    rec = rec[0][:siglen:]
    
    buffer = 30
    rec = rec[buffer*6000::]
    rec = rec[:len(rec)-buffer*6000:].flatten()
    ann = ann[buffer::]
    ann = ann[:len(ann)-buffer:]
    rec = scaler.fit_transform(rec.reshape(-1, 1)).flatten()
    
    np.save(path.join("numpy", f"patient_{i+1}"), rec)
    np.save(path.join("numpy", f"annotation_{i+1}"), ann)
    print(f"Converting patient {i+1} done, signal length: {len(rec)}, total minutes: {len(ann)}")
    
records = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]
for i in range(len(records)):
    rec = wfdb.rdsamp(path.join("database", records[i]))
    ann = wfdb.rdann(path.join("database", records[i]), extension="apn").symbol
    ann = np.array([1 if x == "A" else 0 for x in ann])
    ann.resize(len(ann)-1)
    info = rec[1]
    siglen = len(ann) * 6000
    rec = rec[0][:siglen:]
    
    buffer = 30
    rec = rec[buffer*6000::]
    rec = rec[:len(rec)-buffer*6000:]
    ann = ann[buffer::]
    ann = ann[:len(ann)-buffer:]
    rec = rec.T[3::].flatten()
    rec /= 100;
    
    np.save(path.join("numpy", f"SpO2_patient_{i+1}"), rec)
    np.save(path.join("numpy", f"SpO2_annotation_{i+1}"), ann)
    print(f"Converting SpO2_patient {i+1} done, signal length: {len(rec)}, total minutes: {len(ann)}")