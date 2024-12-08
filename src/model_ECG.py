from data_functions import *
import datetime
from model_functions import *
from data_functions import *
import os

def create_model_raw():
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (64, 3, 0.0, 4),
            (128, 3, 0.0, 4),
        ],
        features = 128,
        name = "ECG_raw",
        dimension = 1,
        show_size = "show_size" in sys.argv,
        only_features_map = True,
    )

def create_model_fft():
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (64, 3, 0.0, 4),
            (128, 3, 0.0, 4),
        ],
        features = 128,
        name = "ECG_fft",
        dimension = 1,
        show_size = "show_size" in sys.argv,
        only_features_map = True,
    )

def create_model(name: str):
    raw_model, _, _ = create_model_raw()
    fft_model, _, _ = create_model_fft()
    
    encoder = layers.concatenate([
        raw_model.output,
        fft_model.output,
    ])

    decoder = layers.Reshape((list(encoder.shape[1::]) + [1]))(encoder)
    decoder = layers.Dropout(rate=0.1)(decoder)
    decoder = layers.Conv1D(filters=64, kernel_size=3, kernel_regularizer=reg.L2())(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.LeakyReLU(negative_slope=0.2)(decoder)
    decoder = layers.MaxPool1D(pool_size=2)(decoder)
    decoder = layers.Flatten()(decoder)
    decoder = layers.Dense(128, activation=layers.LeakyReLU(negative_slope=0.2))(decoder)
    decoder = layers.Dropout(rate=0.1)(decoder)
    decoder = layers.Dense(1, activation="sigmoid")(decoder)
    
    model = Model(
        inputs = [raw_model.input, fft_model.input],
        outputs = decoder,
        name = name 
    )
    
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = [
            metrics.BinaryAccuracy(name = f"threshold_0.{t}",
                                    threshold = t/10) for t in range(1, 10)
        ],
    )

    if "show_size" in sys.argv:
        show_params(model, "ECG_combined")

    return model, encoder

save_path = path.join("res", "model_ECG.keras")
analyzer_path = path.join("res", "analyzer_ECG.keras")

if "epochs" in sys.argv:
    epochs = int(sys.argv[sys.argv.index("epochs")+1])
else:
    epochs = int(input("Please provide a valid number of epochs: "))
batch_size = 64

print("Creating model architecture...")
model, encoder = create_model("ECG_combined")
analyzer = Model(inputs=model.input, outputs=encoder)
original = model.weights

print("Loading data...")

X_raw_train = np.load(path.join("gen_data", "ECG_raw_X_train.npy"))
X_fft_train = np.load(path.join("gen_data", "ECG_fft_X_train.npy"))
y_train = np.load(path.join("gen_data", "ECG_y_train.npy"))

X_raw_test = np.load(path.join("gen_data", "ECG_raw_X_test.npy"))
X_fft_test = np.load(path.join("gen_data", "ECG_fft_X_test.npy"))
y_test = np.load(path.join("gen_data", "ECG_y_test.npy"))

counts = Counter(list(y_train) + list(y_test))
print("Done!")
print(f"Total: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")
print(f"=> Training with {epochs} epochs")

if not "skip_verify" in sys.argv:
    prompt = input("Continue? [y/N]: ")
    if prompt != "y":
        exit()

# callbacks
cb_timer = TimingCallback()
cb_early_stopping = cbk.EarlyStopping(
    patience = 5, 
    restore_best_weights = True,
    start_from_epoch = 100,
)
cb_checkpoint = cbk.ModelCheckpoint(
    save_path, save_best_only = True
)
lr_scheduler = cbk.ReduceLROnPlateau(
    factor = 0.5,
    min_lr = 0.0001,
)

if not "id" in sys.argv:
    name = input("Please provide an id for this section: ")
else:
    name = sys.argv[sys.argv.index("id")+1]

if sys.argv[1] == "std":
    print()
    _s = f"| SECTION {name} |"
    _space = " " * 3
    print(_space + "=" * len(_s), _space + _s, _space + "=" * len(_s), sep="\n")
    now = datetime.datetime.now()
    print("Start at:", now, "\n")
    
    count_train = Counter(y_train)
    count_test = Counter(y_test)
    print(f"=> Train set: Apnea cases [1]: {count_train[1]} - Normal cases [0]: {count_train[0]}")
    print(f"=> Validation set: Apnea cases [1]: {count_test[1]} - Normal cases [0]: {count_test[0]}")

    hist = model.fit([X_raw_train, X_fft_train], 
                        y_train, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        validation_data = ([X_raw_test, X_fft_test], y_test), 
                        callbacks = [
                        cb_timer,
                        cb_early_stopping,
                        cb_checkpoint,
                        lr_scheduler,
                        ])
    t = sum(cb_timer.logs)
    print(f"Total training time: {convert_seconds(t)}")
    analyzer.save(analyzer_path)
    
    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{name}_{key}_ECG")
        np.save(his_path, data)
    print("Saving history done!")
        
if sys.argv[1] == "k_fold":
    X_raw = np.vstack([
        X_raw_train,
        X_raw_test
    ])
    X_fft = np.vstack([
        X_fft_train,
        X_fft_test
    ])
    y = np.hstack([
        y_train,
        y_test
    ])
    
    X_raw, X_fft, y = shuffle(X_raw, X_fft, y, random_state=22022009)
    
    if not "folds" in sys.argv:
        folds = int(input("Please provide an valid number of folds for this section: "))
    else:
        folds = int(sys.argv[sys.argv.index("folds")+1])
    kf = KFold(n_splits=folds)
    
    idx = 0
    scores = []
    
    f = open(path.join("history", f"{name}_k_fold_ECG.txt"), "w")
    
    for train_index, test_index in kf.split(y):
        cb_timer = TimingCallback()
        lr_scheduler = cbk.ReduceLROnPlateau(
            factor = 0.5,
            min_lr = 0.0001,
        )
        cb_early_stopping = cbk.EarlyStopping(
            patience = 5, 
            restore_best_weights = True,
            start_from_epoch = 100,
        )
        idx += 1
        print(f"FOLD {idx}:")
        print(f"FOLD {idx}:", file=f)
        
        X_raw_train, X_raw_test = X_raw[train_index], X_raw[test_index]
        X_fft_train, X_fft_test = X_fft[train_index], X_fft[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        counts_train = Counter(list(y_train.flatten()))
        counts_test = Counter(list(y_test.flatten()))
        print(f"Train set: Apnea cases [1]: {counts_train[1]} - Normal cases [0]: {counts_train[0]}")
        print(f"Test set: Apnea cases [1]: {counts_test[1]} - Normal cases [0]: {counts_test[0]}")
        print(f"Train set: Apnea cases [1]: {counts_train[1]} - Normal cases [0]: {counts_train[0]}", file=f)
        print(f"Test set: Apnea cases [1]: {counts_test[1]} - Normal cases [0]: {counts_test[0]}", file=f)
        
        model.set_weights(original)
        model.fit([X_raw_train, X_fft_train], 
                  y_train, 
                  epochs = epochs, 
                  batch_size = batch_size,
                  verbose = False,
                  callbacks = [
                      cb_timer,
                      lr_scheduler,
                      cb_early_stopping
                  ],
                  validation_data=([X_raw_test, X_fft_test], y_test))
        
        t = sum(cb_timer.logs)
        print(f"Total training time: {convert_seconds(t)}")
        print(f"Total epochs: {len(cb_timer.logs)}")
        print(f"Total epochs: {len(cb_timer.logs)}", file=f)
        
        pred = model.predict([X_raw_test, X_fft_test], verbose=False)
        pred = [np.round(np.squeeze(x)) for x in pred]
        print(classification_report(y_test, pred, target_names=["NO OSA", "OSA"]), file=f)
        cm = confusion_matrix(y_test, pred)
        print("Confusion matrix:\n", cm, file=f)
        
        score = model.evaluate([X_raw_test, X_fft_test], y_test, batch_size=batch_size, verbose=False)[1::]
        scores.append(score)
        for threshold in range(1, 10):
            print(f"Threshold 0.{threshold}: {score[threshold-1]}")
            print(f"Threshold 0.{threshold}: {score[threshold-1]}", file=f)
        
        print()
    
    scores = np.mean(np.array(scores), axis=0)
    print("AVERAGE SCORE")
    print("AVERAGE SCORE", file=f)
    for threshold in range(1, 10):
        print(f"Threshold 0.{threshold}: {scores[threshold-1]}")
        print(f"Threshold 0.{threshold}: {scores[threshold-1]}", file=f)
        
    f.close()