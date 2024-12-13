import numpy as np
from os import path
import os
import sys
import wfdb
import sklearn.preprocessing as prep

print("*** Make sure that you have changed the dataset directoríe name to \"database1\" and \"database2\" before")

### DATABASE1
f = open(path.join("database1", "list"))
records = f.read().splitlines()
f.close()

total_sig_len = 0 
total_minutes = 0

f = open(path.join("gen_data", "ECG-SpO2.txt"), "w")

SpO2_records = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]
raw_SpO2_records = list(map(lambda x: x[:-1:], SpO2_records))
scaler = prep.MinMaxScaler()
for i in range(len(records)):
    if records[i][0] == "x":
        continue

    if records[i] in raw_SpO2_records:
        f.write(str(i+1) + " \n")

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

f.close()

records = SpO2_records
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
    rec /= 100
    
    np.save(path.join("numpy", f"SpO2_patient_{i+1}"), rec)
    np.save(path.join("numpy", f"SpO2_annotation_{i+1}"), ann)
    print(f"Converting SpO2_patient {i+1} done, signal length: {len(rec)}, total minutes: {len(ann)}")
    
    
### DATABASE2
if not "extra" in sys.argv:
    exit()
f = open(path.join("database2", "RECORDS"))
records = f.read().splitlines()
f.close()

st = 35

osa = ["H", "HA", "OA", "X", "CA", "CAA"]
def check_osa(x: str) -> bool:
    for ann in osa:
        if ann in x:
            return True
    return False

for i, record in enumerate(records):
    record = path.join("database2", record)
    rec = wfdb.rdrecord(record).p_signal.T[0].flatten()
    ann = wfdb.rdann(record, 'st').aux_note
    label = []
    for idx, note in enumerate(ann):
        if check_osa(note):
            label.append(1)
        else:
            label.append(0)
    ann = np.array(label)
    np.save(path.join("numpy", f"patient_{st+i+1}"), rec)
    np.save(path.join("numpy", f"annotation_{st+i+1}"), ann)
    print(f"Converting patient {st+i+1} done, signal length: {len(rec)}, total minutes: {len(ann)*2}")
    