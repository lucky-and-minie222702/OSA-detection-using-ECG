from typing import Counter
import numpy as np
from os import path
import sys
from sklearn.utils import shuffle
from librosa.feature import mfcc, delta
import sklearn.preprocessing as prep
from data_functions import *
import math

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
print(f"Total signal's length: {total_sig_len[0]}\n")
print("** SpO2 ***")
print(f"Total sleep minutes: {total_minutes[1]}")
print(f"Total signal's length: {total_sig_len[1]}\n")

if sys.argv[1] == "percentage":
    print(f"{"*"*10} ECG patients {"*"*10}\n")
    for idx, ann in enumerate(ann_ECG):
        counter = Counter(ann)
        ann_len = len(ann)
        print(f"{"="*10} patient {idx+1:^4} {"="*10}")
        print(f"| Apnea: {counter[1]:>6} | Normal: {counter[0]:>6} |")
        print(f"| Apnea: {round(counter[1] / ann_len * 100, 2):>5}% | Normal: {round(counter[0] / ann_len * 100, 2):>5}% |")
        print("=" * 34, "\n")
        
    print()
    print(f"{"*"*10} SpO2 patients {"*"*10}\n")
    for idx, ann in enumerate(ann_SpO2):
        counter = Counter(ann)
        ann_len = len(ann)
        print(f"{"="*10} patient {idx+1:^4} {"="*10}")
        print(f"| Apnea: {counter[1]:>6} | Normal: {counter[0]:>6} |")
        print(f"| Apnea: {round(counter[1] / ann_len * 100, 2):>5}% | Normal: {round(counter[0] / ann_len * 100, 2):>5}% |")
        print("=" * 34, "\n")
    print("Done!")
    
if sys.argv[1] == "plot":
    print(f"ECG patients")
    for idx, ann in enumerate(ann_ECG):
        ann = np.array(ann)[:len(ann) - (len(ann) % 100):]
        ann = np.array(np.split(ann, 100))
        plot = np.round(np.mean(ann, axis=1))
        _plt = []
        _s = "".join(["X" if p == 1 else "-" for p in plot])
        print(f"Patient {idx+1} :")
        print(_s)
        
    print()
    print(f"SpO2 patients")
    for idx, ann in enumerate(ann_SpO2):
        ann = np.array(ann)[:len(ann) - (len(ann) % 100):]
        ann = np.array(np.split(ann, 100))
        plot = np.round(np.mean(ann, axis=1))
        _s = "".join(["X" if p == 1 else "-" for p in plot])
        print(f"Patient {idx+1}:")
        print(_s)

    print("Done!")

stdev_SpO2 = [[], []]
mean_SpO2 = [[], []]
if sys.argv == "statistics_SpO2":
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
    
if sys.argv[1] == "merge":
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


if sys.argv[1] == "save_features":
    _s = "a_" if "augmented" in sys.argv else ""
    print("Extracting ECG...")
    X_0 = np.load(path.join("gen_data", f"{_s}ECG_normal.npy"))
    X_1 = np.load(path.join("gen_data", f"{_s}ECG_apnea.npy"))
    print("Extracting normal patients...")
    X_0_fft, X_0_psd = extract_features(X_0, sampling_rate=100, verbose=True)
    print("Extracting apnea patients...")
    X_1_fft, X_1_psd = extract_features(X_1, sampling_rate=100, verbose=True)
    print("Exporting...")
    np.save(path.join("gen_data", "fft_ECG_normal"), X_0_fft)
    np.save(path.join("gen_data", "fft_ECG_apnea"), X_1_fft)
    np.save(path.join("gen_data", "psd_ECG_normal"), X_0_psd)
    np.save(path.join("gen_data", "psd_ECG_apnea"), X_1_psd)
    print("Done!")
    
if sys.argv[1] == "save_stats":
    _s = "a_" if "augmented" in sys.argv else ""
    print("Calculating SpO2...")
    X_0 = np.load(path.join("gen_data", f"{_s}SpO2_normal.npy"))
    X_1 = np.load(path.join("gen_data", f"{_s}SpO2_apnea.npy"))
    print("Extracting normal patients...")
    X_0, keys = extract_stats(X_0, sampling_rate=100, verbose=True)
    print("Extracting apnea patients...")
    X_1, _ = extract_stats(X_1, sampling_rate=100, verbose=True)
    print("Exporting...")
    np.save(path.join("gen_data", "s_SpO2_normal"), X_0)
    np.save(path.join("gen_data", "s_SpO2_apnea"), X_1)
    
    f = open(path.join("gen_data", "stats_keys.txt"), "w")
    for k in keys:
        f.write(k + "\n")
    f.close()
    
    print("Done!")
    
if sys.argv[1] == "pair":
    print("Loading data...")
    p_list = open(path.join("gen_data", "ECG-SpO2.txt"), "r").readlines()
    p_list = list(map(lambda x: int(x), p_list))
    X_ECG, _ = get_patients_ECG(p_list)
    y, _ = get_patients_SpO2(range(1, 9))
    if "augment" in sys.argv:
        print("Augmenting...")
        X_ECG = np.vstack(
            [X_ECG, np.flip(X_ECG, axis=1)]
        ) 
        y = np.vstack(
            [y, np.flip(y)]
        ) 
    print("Extracing SpO2 statistics...")
    y, _ = extract_stats(y, sampling_rate=100, save_scaler=True, verbose=True)
    np.save(path.join("gen_data", "ECG-pair"), X_ECG)
    np.save(path.join("gen_data", "y-pair"), y)
    print("Done!")
    
if sys.argv[1] == "augment":
    if "SpO2" in sys.argv:
        print("Augmenting SpO2...")
        X_0 = np.load(path.join("gen_data", "SpO2_normal.npy"))
        X_1 = np.load(path.join("gen_data", "SpO2_apnea.npy"))
        
        a_X_0 = np.vstack(
            [X_0, np.flip(X_0, axis=1)],
        )
        a_X_1 = np.vstack(
            [X_1, np.flip(X_1, axis=1)],
        )
        
        a_X_0 = np.vstack(
            [X_0, X_0 + np.random.normal(0, 0.1, X_0.shape)],
        )
        a_X_1 = np.vstack(
            [X_1, X_1 + np.random.normal(0, 0.1, X_1.shape)],
        )
        
        np.save(path.join("gen_data", "a_SpO2_apnea.npy"), a_X_1)
        np.save(path.join("gen_data", "a_SpO2_normal.npy"), a_X_0)
        print("Done!")
        
    if "ECG" in sys.argv:
        print("Augmenting ECG...")
        X_0 = np.load(path.join("gen_data", "ECG_normal.npy"))
        X_1 = np.load(path.join("gen_data", "ECG_apnea.npy"))
        
        a_X_0 = np.vstack(
            [X_0, np.flip(X_0, axis=1)],
        )
        a_X_1 = np.vstack(
            [X_1, np.flip(X_1, axis=1)],
        )
        
        a_X_0 = np.vstack(
            [X_0, X_0 + np.random.normal(0, 0.0075, X_0.shape)],
        )
        a_X_1 = np.vstack(
            [X_1, X_1 + np.random.normal(0, 0.0075, X_1.shape)],
        )
        
        np.save(path.join("gen_data", "a_ECG_normal.npy"), a_X_0)
        np.save(path.join("gen_data", "a_ECG_apnea.npy"), a_X_1)
        print("Done!")