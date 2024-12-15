from typing import Counter
import numpy as np
from os import path
import sys
from sklearn.utils import shuffle
from librosa.feature import mfcc, delta
import sklearn.preprocessing as prep
from data_functions import *
import math
from sklearn.model_selection import train_test_split
import pickle

f = open(path.join("database1", "list"))
records = f.read().splitlines()

total_sig_len = [0, 0] 
total_minutes = [0, 0]

ann_ECG = []
ann_SpO2 = []

for i in range(len(records)):
    if records[i][0] == "x":
        continue
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
    print(f"{'*'*10} ECG patients {'*'*10}\n")
    for idx, ann in enumerate(ann_ECG):
        counter = Counter(ann)
        ann_len = len(ann)
        print(f"{'='*10} patient {idx+1:^4} {'='*10}")
        print(f"| Apnea: {counter[1]:>6} | Normal: {counter[0]:>6} |")
        print(f"| Apnea: {round(counter[1] / ann_len * 100, 2):>5}% | Normal: {round(counter[0] / ann_len * 100, 2):>5}% |")
        print("=" * 34, "\n")
        
    print()
    print(f"{'*'*10} SpO2 patients {'*'*10}\n")
    for idx, ann in enumerate(ann_SpO2):
        counter = Counter(ann)
        ann_len = len(ann)
        print(f"{'='*10} patient {idx+1:^4} {'='*10}")
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

if sys.argv[1] == "merge":
    merged_X = [[], []]
    if "ECG" in sys.argv:
        print("Merging ECG...")
        X, y = get_patients_ECG((range(1, 36)))
        
        scaler = prep.MinMaxScaler()
        X = scaler.fit_transform(X.flatten().reshape(-1, 1)).reshape(-1, 6000)
        joblib.dump(scaler, path.join("res", "ECG_scaler.scaler"))
        
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
    
    merged_X = [[], []]
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
    print("Done!")

# Legend
if sys.argv[1] == "save_features":
    print("Extracting ECG...")
    X_0 = np.load(path.join("gen_data", "a_ECG_normal.npy"))
    X_1 = np.load(path.join("gen_data", "a_ECG_apnea.npy"))
    print("Extracting normal patients...")
    X_0_fft = Legend.extract_features2(X_0, sampling_rate=100, verbose=True)
    print("Extracting apnea patients...")
    X_1_fft = Legend.extract_features2(X_1, sampling_rate=100, verbose=True)
    print("Exporting...")
    np.save(path.join("gen_data", "fft_ECG_normal"), X_0_fft)
    np.save(path.join("gen_data", "fft_ECG_apnea"), X_1_fft)
    print("Done!")

if sys.argv[1] == "save_stats":
    if "SpO2" in sys.argv:
        print("Calculating SpO2...")
        X_0 = np.load(path.join("gen_data", "a_SpO2_normal.npy"))
        X_1 = np.load(path.join("gen_data", "a_SpO2_apnea.npy"))
        
        print("Initializing scaler...")
        tmp = np.vstack([X_0, X_1])
        _, _ = Legend.extract_stats(tmp, sampling_rate=100, save_scaler=True, name="SpO2")
        print("Done!")
        
        print("Extracting normal patients...")
        X_0, keys = Legend.extract_stats(X_0, sampling_rate=100, verbose=True)
        print("Extracting apnea patients...")
        X_1, _ = Legend.extract_stats(X_1, sampling_rate=100, verbose=True)
        print("Exporting...")
        np.save(path.join("gen_data", "s_SpO2_normal"), X_0)
        np.save(path.join("gen_data", "s_SpO2_apnea"), X_1)
        
        f = open(path.join("gen_data", "stats_keys.txt"), "w")
        for k in keys:
            f.write(k + "\n")
        f.close()
        
        print("Done!")
    if "ECG" in sys.argv:
        print("Calculating ECG...")
        X_0 = np.load(path.join("gen_data", "a_ECG_normal.npy"))
        X_1 = np.load(path.join("gen_data", "a_ECG_apnea.npy"))
        
        print("Initializing scaler...")
        tmp = np.vstack([X_0, X_1])
        _, _ = Legend.extract_stats(tmp, sampling_rate=100, save_scaler=True, name="ECG")
        print("Done!")
        
        print("Extracting normal patients...")
        X_0, keys = Legend.extract_stats(X_0, sampling_rate=100, verbose=True)
        print("Extracting apnea patients...")
        X_1, _ = Legend.extract_stats(X_1, sampling_rate=100, verbose=True)
        print("Exporting...")
        np.save(path.join("gen_data", "s_ECG_normal"), X_0)
        np.save(path.join("gen_data", "s_ECG_apnea"), X_1)
        
        f = open(path.join("gen_data", "stats_keys.txt"), "w")
        for k in keys:
            f.write(k + "\n")
        f.close()
        
        print("Done!")
    
if sys.argv[1] == "pair":
    print("Loading data...")
    p_list = [1, 2, 3, 4, 21, 26, 27, 28]
    X_ECG, _ = get_patients_ECG(p_list)
    X_SpO2, _ =get_patients_SpO2(range(1, 9))
    
    ideal = min(len(X_ECG), len(X_SpO2))
    X_ECG = X_ECG[:ideal:]
    X_SpO2 = X_SpO2[:ideal:]
    
    print("Calculating ECG...")
    print("Calculating SpO2...")
    
    print("Pairing...")
    X_pair = np.stack([
        X_ECG,
        X_SpO2,
    ], axis=0)
    
    np.save(path.join("gen_data", "pair_ECG_SpO2"), X_pair)
    print("Done!")
    
if sys.argv[1] == "augment":
    overlap_size = 3000

    if "SpO2" in sys.argv:
        print("Augmenting SpO2...")
        X_0 = np.load(path.join("gen_data", "SpO2_normal.npy"))
        X_1 = np.load(path.join("gen_data", "SpO2_apnea.npy"))
        
        a_X_0 = np.vstack(
            [X_0, X_0 + np.random.normal(0, 0.01, X_0.shape)],
        )
        a_X_1 = np.vstack(
            [X_1, X_1 + np.random.normal(0, 0.01, X_1.shape)],
        )
        
        if "overlap" in sys.argv:
            tmp = a_X_0.flatten()[overlap_size:len(a_X_0)*6000-overlap_size:]
            tmp = np.array(np.split(tmp, len(tmp) // 6000))
            a_X_0 = np.vstack(
                [a_X_0, tmp]
            )
            tmp = a_X_1.flatten()[overlap_size:len(a_X_1)*6000-overlap_size:]
            tmp = np.array(np.split(tmp, len(tmp) // 6000))
            a_X_1 = np.vstack(
                [a_X_1, tmp]
            )
        
        np.save(path.join("gen_data", "a_SpO2_apnea.npy"), a_X_1)
        np.save(path.join("gen_data", "a_SpO2_normal.npy"), a_X_0)
        print("Done!")
        
    if "ECG" in sys.argv:
        print("Augmenting ECG...")
        X_0 = np.load(path.join("gen_data", "ECG_normal.npy"))
        X_1 = np.load(path.join("gen_data", "ECG_apnea.npy"))

        a_X_0 = np.vstack(
            [X_0, X_0 + np.random.normal(0, 0.001, X_0.shape)],
        )
        a_X_1 = np.vstack(
            [X_1, X_1 + np.random.normal(0, 0.001, X_1.shape)],
        )
        
        a_X_0 = np.vstack(
            [a_X_0, a_X_0 - 0.01, a_X_0 + 0.01],
        )
        a_X_1 = np.vstack(
            [a_X_1, a_X_1 - 0.01, a_X_1 + 0.01],
        )
        
        np.save(path.join("gen_data", "a_ECG_normal.npy"), a_X_0)
        np.save(path.join("gen_data", "a_ECG_apnea.npy"), a_X_1)
        print("Done!")
        
if sys.argv[1] == "split_dataset":
    _s = "s_" if "stats" in sys.argv else "a_" # Legend
    if "SpO2" in sys.argv:
        print("Splitting SpO2...")
        a, b = np.load(path.join("gen_data", f"{_s}SpO2_normal.npy")), np.load(path.join("gen_data", f"{_s}SpO2_apnea.npy"))
        X = np.vstack([
            a, b
        ])
        y = np.array([[0] * len(a) + [1] * len(b)]).flatten()
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(10000000))
        np.save(path.join("gen_data", "SpO2_X_train"), X[train_indices])
        np.save(path.join("gen_data", "SpO2_y_train"), y[train_indices])
        np.save(path.join("gen_data", "SpO2_X_test"), X[test_indices])
        np.save(path.join("gen_data", "SpO2_y_test"), y[test_indices])
        print("Done!")

    if "ECG" in sys.argv:
        print("Splitting ECG...")
        a, b = np.load(path.join("gen_data", f"{_s}ECG_normal.npy")), np.load(path.join("gen_data", f"{_s}ECG_apnea.npy"))
        X = np.vstack([
            a, b
        ])
        y = np.array([[0] * len(a) + [1] * len(b)]).flatten()
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=np.random.randint(10000000))
        np.save(path.join("gen_data", "ECG_X_train"), X[train_indices])
        np.save(path.join("gen_data", "ECG_y_train"), y[train_indices])
        np.save(path.join("gen_data", "ECG_X_test"), X[test_indices])
        np.save(path.join("gen_data", "ECG_y_test"), y[test_indices])
        print("Done!")
        
if sys.argv[1] == "chop":
    division = 500
    print("Chopping...")
    X_0 = np.load(path.join("gen_data", "ECG_normal.npy")).flatten()
    X_1 = np.load(path.join("gen_data", "ECG_apnea.npy")).flatten()
    X_0 = np.array(np.split(X_0, len(X_0) // division)).squeeze()
    X_1 = np.array(np.split(X_1, len(X_1) // division)).squeeze()
    np.save(path.join("gen_data", "ECG_normal.npy"), X_0)
    np.save(path.join("gen_data", "ECG_apnea.npy"), X_1)
    print("Done!")

# if sys.argv[1] == "add_extra":
#     X, y = get_patients_ECG(range(36, 54))
#     print(X.shape, y.shape)
#     pass