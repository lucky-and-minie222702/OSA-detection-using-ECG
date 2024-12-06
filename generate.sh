python src/preprocess.py
python src/patients_action.py merge ECG SpO2
python src/patients_action.py augment ECG SpO2
python src/patients_action.py save_features augmented
python src/patients_action.py save_stats augmented
python src/patients_action.py split_dataset ECG SpO2 augmented