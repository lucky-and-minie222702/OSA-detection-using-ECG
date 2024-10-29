from typing import Counter
import numpy as np
from os import path
import sys
from sklearn.utils import shuffle
from librosa.feature import mfcc, delta
import sklearn.preprocessing as prep
from data_functions import *

f = open(path.join("database", "list"))
records = f.read().splitlines()

total_sig_len = [0, 0] 
total_minutes = [0, 0]

ann_ECG = []
ann_SpO2 = []

for i in range(len(records)):
    if records[i][0] == "x":
        break
    ann = np.load(path.join("numpy", f"annotation_{i+1}.npy"))
    ann_ECG.append(ann)
    total_minutes[0] += len(ann)
    total_sig_len[0] += len(ann) * 6000
    
records = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]
records_ECG = [1, 2, 3, 4, 21, 26, 27, 28]
for i in range(len(records)):
    ann = np.load(path.join("numpy", f"SpO2_annotation_{i+1}.npy"))
    ann_SpO2.append(ann)
    total_minutes[1] += len(ann)
    total_sig_len[1] += len(ann) * 6000
    
print("** ECG ***")
print(f"Total sleep minutes: {total_minutes[0]}")
print(f"Total signal's length: {total_sig_len[0]}")
print("** SpO2 ***")
print(f"Total sleep minutes: {total_minutes[1]}")
print(f"Total signal's length: {total_sig_len[1]}")

if "percentage" in sys.argv:
    print(f"{"*"*10} ECG patients {"*"*10}\n")
    for idx, ann in enumerate(ann_ECG):
        counter = Counter(ann)
        ann_len = len(ann)
        print(f"{"-"*10} patient {idx+1:^4} {"-"*10}")
        print(f"| Apnea: {counter[1]:>6} | Normal: {counter[0]:>6} |")
        print(f"| Apnea: {round(counter[1] / ann_len * 100, 2):>5}% | Normal: {round(counter[0] / ann_len * 100, 2):>5}% |")
    print()
    print(f"{"*"*10} SpO2 patients {"*"*10}\n")
    for idx, ann in enumerate(ann_ECG):
        counter = Counter(ann)
        ann_len = len(ann)
        print(f"{"-"*10} patient {idx+1:^4} {"-"*10}")
        print(f"| Apnea: {counter[1]:>6} | Normal: {counter[0]:>6} |")
        print(f"| Apnea: {round(counter[1] / ann_len * 100, 2):>5}% | Normal: {round(counter[0] / ann_len * 100, 2):>5}% |")
    print("Done!")

stdev_SpO2 = [[], []]
mean_SpO2 = [[], []]
if "statistics_SpO2" in sys.argv:
    X, y = get_patients_SpO2(range(1, 9))
    for i in range(len(y)):
        stdev_SpO2[y[i]].append(np.std(X[i], axis=0))
        mean_SpO2[y[i]].append(np.mean(X[i], axis=0))

    if "save_stat" in sys.argv:
        np.save(path.join("stat", "mean_SpO2_normal"), np.array(mean_SpO2[0]))
        np.save(path.join("stat", "mean_SpO2_apnea"), np.array(mean_SpO2[1]))
        np.save(path.join("stat", "stdev_SpO2_normal"), np.array(stdev_SpO2[0]))
        np.save(path.join("stat", "stdev_SpO2_apnea"), np.array(stdev_SpO2[1]))

    mean_SpO2[0] = np.mean(mean_SpO2[0], axis=0)
    mean_SpO2[1] = np.mean(mean_SpO2[1], axis=0)
    print("*** Mean ***")
    print("Normal:", mean_SpO2[0])
    print("Apnea:", mean_SpO2[1])

    stdev_SpO2[0] = np.mean(stdev_SpO2[0], axis=0)
    stdev_SpO2[1] = np.mean(stdev_SpO2[1], axis=0)
    print("*** Standard deviation ***")
    print("Normal:", stdev_SpO2[0])
    print("Apnea:", stdev_SpO2[1])
    X, y = [], []

    print("Done!")
    
if "merge" in sys.argv:
    merged_X = [[], []]
    if "ECG" in sys.argv:
        print("Merging ECG...")
        X, y = get_patients_ECG((range(1, 36)))
        counts = Counter(y)
        ideal = min(counts[0], counts[1])
        for i in range(len(y)):
            merged_X[y[i]].append(X[i])

        merged_X[0] = merged_X[0][:ideal:]
        merged_X[1] = merged_X[1][:ideal:]
        merged_X = np.array(merged_X)
        
        np.save(path.join("gen_data", "ECG_normal"), merged_X[0])
        np.save(path.join("gen_data", "ECG_apnea"), merged_X[1])

        print(f"Apnea: {len(merged_X[1])} - Normal: {len(merged_X[0])}")
        X, y, merged_X = [], [], []
    
    if "SpO2" in sys.argv:
        print("Merging SpO2...")
        X, y = get_patients_SpO2((range(1, 9)))
        counts = Counter(y)
        ideal = min(counts[0], counts[1])
        for i in range(len(y)):
            merged_X[y[i]].append(X[i])

        merged_X[0] = merged_X[0][:ideal:]
        merged_X[1] = merged_X[1][:ideal:]
        merged_X = np.array(merged_X)
        
        np.save(path.join("gen_data", "SpO2_normal"), merged_X[0])
        np.save(path.join("gen_data", "SpO2_apnea"), merged_X[1])

        print(f"Apnea: {len(merged_X[1])} - Normal: {len(merged_X[0])}")
        X, y, merged_X = [], [], [], [], []
        print("Done!")
        
if "create_pair_data" in sys.argv:
    X_ECG, y = get_patients_ECG(records_ECG)
    X_SpO2, _ = get_patients_SpO2(range(1, 9))
    X_pair = np.stack([
        X_ECG,
        X_SpO2,
    ], axis=2)
    X_pair, y = shuffle(X_pair, y, random_state=27022009)
    np.save(path.join("gen_data", "rec_pair_data"), X_pair)
    np.save(path.join("gen_data", "ann_pair_data"), y)
    print("Done!")

if "save_features":
    X_0 = np.load(path.join("gen_data", "ECG_normal.npy"))
    X_1 = np.load(path.join("gen_data", "ECG_apnea.npy"))
    print("Extracting normal patients...")
    X_0 = feature_extract(X_0, True)
    print("Extracting Apnea patients...")
    X_1 = feature_extract(X_1, True)
    print("Exporting...")
    np.save(path.join("gen_data", "f_ECG_normal"), X_0)
    np.save(path.join("gen_data", "f_ECG_apnea"), X_1)
    print("Done!")