import datetime
from model_functions import *
from data_functions import *
import os

def create_model(name: str):
    inp = layers.Input(shape=(3,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = Model(
        inputs = inp,
        outputs = out,
    )
    
    model.compile(
        optimizer = "Adam",
        loss = "binary_crossentropy",
        metrics = [
            metrics.BinaryAccuracy(name = f"threshold_0.{t}",
                                    threshold = t/10) for t in range(1, 10)
        ],
    )

    if "show_size" in sys.argv:
        show_params(model, name)
        
    return model

save_path = path.join("res", "model_SpO2.weights.h5")

if "epochs" in sys.argv:
    epochs = int(sys.argv[sys.argv.index("epochs")+1])
else:
    epochs = int(input("Please provide a valid number of epochs: "))
batch_size = 64
es_ep = 10

print("Creating model architecture...")
model = create_model("SpO2_raw")
original = model.weights

print("Loading data...")

X_train = np.load(path.join("gen_data", "SpO2_X_train.npy"))
y_train = np.load(path.join("gen_data", "SpO2_y_train.npy"))

X_test = np.load(path.join("gen_data", "SpO2_X_test.npy"))
y_test = np.load(path.join("gen_data", "SpO2_y_test.npy"))

counts = Counter(list(y_train) + list(y_test))
print("Done!")
print(f"Total: Apnea cases [1]: {counts[1]} - Normal cases [0]: {counts[0]}")
print(f"Training with {epochs} epochs limit!")

if not "skip_verify" in sys.argv:
    prompt = input("Continue? [y/N]: ")
    if prompt != "y":
        exit()

# callbacks
cb_timer = TimingCallback()
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = es_ep,
)
cb_checkpoint = cbk.ModelCheckpoint(
    save_path, 
    save_best_only=True,
    save_weights_only=True,
)
lr_scheduler = cbk.ReduceLROnPlateau(
    factor = 0.5,
    min_lr = 0.000001,
    patience = 5,
)
cb_forget = DynamicWeightSparsification(
    sparsity_target = 0.01,
    layer_names = ["final"],
    end_from_epoch = 10,
)
cb_weight_memory = WeightMemoryMechanism(
    patience = 3,
    start_from_epoch = es_ep - 10,
)   

if not "id" in sys.argv:
    name = input("Please provide an id for this section: ")
else:
    name = sys.argv[sys.argv.index("id")+1]

print()
_s = f"| SECTION {name} |"
_space = " " * 3
print(_space + "=" * len(_s), _space + _s, _space + "=" * len(_s), sep="\n")
now = datetime.datetime.now()
print("Start at:", now, "\n")

if sys.argv[1] == "std":    
    if "threshold" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("threshold")+1])
    else:
        threshold = float(input("Please provide a valid threshold: "))
    
    count_train = Counter(y_train)
    count_test = Counter(y_test)
    print(f"Train set: Apnea cases [1]: {count_train[1]} - Normal cases [0]: {count_train[0]}")
    print(f"Validation set: Apnea cases [1]: {count_test[1]} - Normal cases [0]: {count_test[0]}")

    hist = model.fit(
                        X_train, 
                        y_train, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        validation_split = 0.2,
                        callbacks = [
                            cb_timer,
                            cb_early_stopping,
                            cb_checkpoint,
                            lr_scheduler,
                        ])
    t = sum(cb_timer.logs)
    
    f = open(path.join("history", "SpO2_train.txt"), "w")
    
    print(f"Total training time: {convert_seconds(t)}")
    print(f"Total epochs: {len(cb_timer.logs)}")
    print(f"Total epochs: {len(cb_timer.logs)}", file=f)
    
    pred = model.predict(X_test, verbose=False, batch_size=batch_size*4)
    arr = np.array([np.squeeze(x) for x in pred])
    pred =  np.where(arr % 1 >= threshold, np.ceil(arr), np.floor(arr))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)
    print("Confusion matrix:\n", cm, file=f)
    
    score = model.evaluate(X_test, y_test, batch_size=batch_size*4, verbose=False)[1::]
    for threshold in range(1, 10):
        print(f"Threshold 0.{threshold}: {score[threshold-1]}")
        print(f"Threshold 0.{threshold}: {score[threshold-1]}", file=f)
    
    f.close()
    print()
    
    for key, value in hist.history.items():
        data = np.array(value)
        his_path = path.join("history", f"{name}_{key}_SpO2")
        np.save(his_path, data)

    print("Saving history done!")
        
if sys.argv[1] == "k_fold":
    X = np.vstack([
        X_train,
        X_test
    ])
    y = np.hstack([
        y_train,
        y_test
    ])
    
    X, y = shuffle(X, y, random_state=np.random.randint(10000000))
    
    if not "folds" in sys.argv:
        folds = int(input("Please provide an valid number of folds for this section: "))
    else:
        folds = int(sys.argv[sys.argv.index("folds")+1])
    kf = KFold(n_splits=folds, shuffle=True)
    
    idx = 0
    scores = []
    
    f = open(path.join("history", f"{name}_k_fold_SpO2.txt"), "w")
    
    for train_index, test_index in kf.split(y):
        cb_timer = TimingCallback()
        idx += 1
        print(f"FOLD {idx}:")
        print(f"FOLD {idx}:", file=f)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        counts_train = Counter(list(y_train.flatten()))
        counts_test = Counter(list(y_test.flatten()))
        print(f"Train set: Apnea cases [1]: {counts_train[1]} - Normal cases [0]: {counts_train[0]}")
        print(f"Test set: Apnea cases [1]: {counts_test[1]} - Normal cases [0]: {counts_test[0]}")
        print(f"Train set: Apnea cases [1]: {counts_train[1]} - Normal cases [0]: {counts_train[0]}", file=f)
        print(f"Test set: Apnea cases [1]: {counts_test[1]} - Normal cases [0]: {counts_test[0]}", file=f)

        model.set_weights(original)
        model.fit(
                    X_train, 
                    y_train, 
                    epochs = epochs, 
                    batch_size = batch_size,
                    verbose = False,
                    callbacks = [
                        cb_timer,
                        lr_scheduler,
                        cb_early_stopping,
                        EpochProgressCallback(),
                    ],
                    validation_split = 0.2,
                    )
        
        t = sum(cb_timer.logs)
        print(f"Total training time: {convert_seconds(t)}")
        print(f"Total epochs: {len(cb_timer.logs)}")
        print(f"Total epochs: {len(cb_timer.logs)}", file=f)
        
        score = model.evaluate(X_test, y_test, batch_size=batch_size*4, verbose=False)[1::]
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