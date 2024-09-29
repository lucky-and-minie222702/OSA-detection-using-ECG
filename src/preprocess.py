import numpy as np
from os import path
import os
import sys
import argparse
import wfdb
from librosa.feature import mfcc

if not path.isdir("numpy"):
    os.makedirs("numpy")

print("*** Make sure that you have changed the dataset directory name to \"database\" before")

f = open(path.join("database", "list"))
records = f.read().splitlines()

total_sig_len = 0 
total_minutes = 0

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
    
    buffer = 10
    rec = rec[buffer*6000::]
    rec = rec[:len(rec)-buffer*6000:]
    ann = ann[buffer::]
    ann = ann[:len(ann)-buffer:]
    
    np.save(path.join("numpy", f"patient_{i+1}"), rec)
    np.save(path.join("numpy", f"annotation_{i+1}"), ann)
    print(f"Converting patient {i+1} done, signal length: {len(rec)}, total minutes: {len(ann)}")