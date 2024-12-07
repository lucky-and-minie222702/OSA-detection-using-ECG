from data_functions import *
import datetime
from model_functions import *
from data_functions import *
import os

def create_model():
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (32, 5, 0.01),
            (64, 5, 0.0),
            (128, 3, 0.0),
            (256, 3, 0.0),
        ],
        decoder_structures = [
            (256, 0.001),
            (128, 0.0),
        ],
        features = 256,
        name = "raw_SpO2",
        dimension = 1,
        show_size = True,
        compile = True
    )

save_path = path.join("res", "model_SpO2.keras")
analyzer_path = path.join("res", "analyzer_SpO2.keras")

if "epochs" in sys.argv:
    epochs = int(sys.argv[sys.argv.index("epochs")+1])
else:
    epochs = int(input("Please provide a valid number of epochs: "))
batch_size = 64

print("Creating model architecture...")
model, encoder, _ = create_model()
analyzer = Model(inputs=model.input, outputs=encoder)
original = model.weights

print("Loading data...")

X_train = np.load(path.join("gen_data", "SpO2_X_train.npy"))
y_train = np.load(path.join("gen_data", "SpO2_y_train.npy"))

X_test = np.load(path.join("gen_data", "SpO2_X_test.npy"))
y_test = np.load(path.join("gen_data", "SpO2_y_test.npy"))

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
    save_path, save_best_only=True
)
lr_scheduler = cbk.ReduceLROnPlateau(
    factor = 0.5,
    min_lr = 0.0001,
)

if sys.argv[1] == "std":
    if "build" in sys.argv:
        if not "id" in sys.argv:
            id = input("Please provide an id for this section: ")
        else:
            id = sys.argv[sys.argv.index("id")+1]
    print()
    _s = f"| SECTION {id} |"
    _space = " " * 3
    print(_space + "=" * len(_s), _space + _s, _space + "=" * len(_s), sep="\n")
    now = datetime.datetime.now()
    print("Start at:", now, "\n")
    
    val_split = 0.2
    
    count_train = Counter(y_train)
    count_test = Counter(y_test)
    print(f"=> Train set: Apnea cases [1]: {count_train[1]} - Normal cases [0]: {count_train[0]}")
    print(f"=> Test set: Apnea cases [1]: {count_test[1]} - Normal cases [0]: {count_test[0]}")
    print(f"=> Validation set: Apnea cases [1]: {int(count_train[1]*val_split)} - Normal cases [0]: {int(count_train[0]*val_split)}")

    if "build" in sys.argv:
        hist = model.fit(X_train, 
                         y_train, 
                         epochs = epochs, 
                         batch_size = batch_size, 
                         validation_split = val_split, 
                         callbacks = [
                            cb_timer,
                            cb_early_stopping,
                            cb_checkpoint,
                            lr_scheduler,
                         ])
        t = sum(cb_timer.logs)
        print(f"Total training time: {convert_seconds(t)}")
        analyzer.save(analyzer_path)
    elif "test" in sys.argv:
        model = load_model(save_path)
    print("Evaluating...")
    pred = model.predict(X_test, verbose=False)
    pred = [np.round(np.squeeze(x)) for x in pred]
    f = open(path.join("history", f"{id}_result_SpO2.txt"), "w")
    print(classification_report(y_test, pred, target_names=["NO OSA", "OSA"]), file=f)
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm, file=f)
    names = ["loss"]
    names += [ f"threshold_0.{t}" for t in range(1, 10) ]
    results = model.evaluate(X_test, y_test, verbose=False)
    print("\nLoss and metrics", file=f)
    for idx in range(10):
        print(names[idx], ":", results[idx], file=f)
    f.close() 
    
    if "build" in sys.argv:
        for key, value in hist.history.items():
            data = np.array(value)
            his_path = path.join("history", f"{id}_{key}_SpO2")
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
    
    if not "folds" in sys.argv:
        folds = int(input("Please provide an valid number of folds for this section: "))
    else:
        folds = int(sys.argv[sys.argv.index("folds")+1])
    kf = KFold(n_splits=folds)
    
    idx = 0
    scores = []
    
    f = open(path.join("history", f"{id}_k_fold_SpO2.txt"), "w")
    
    for train_index, test_index in kf.split(X):
        cb_timer = TimingCallback()
        lr_scheduler = cbk.ReduceLROnPlateau(
            factor = 0.5,
            min_lr = 0.0001,
        )
        idx += 1
        print(f"FOlD {idx}:")
        print(f"FOlD {idx}:", file=f)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.set_weights(original)
        model.fit(X_train, 
                  y_train, 
                  epochs = epochs, 
                  batch_size = batch_size,
                  verbose = False,
                  callbacks = [
                      cb_timer,
                      lr_scheduler
                  ])
        
        t = sum(cb_timer.logs)
        print(f"Total training time: {convert_seconds(t)}")

        score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=False)[1::]
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