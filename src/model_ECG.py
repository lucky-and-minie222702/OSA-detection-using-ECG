from data_functions import *
import datetime
from model_functions import *
from data_functions import *
import os

def create_model_raw():
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (32, 3, 0.0),
            (64, 3, 0.0),
            (128, 3, 0.0),
            (256, 3, 0.0),
            (512, 3, 0.0),
        ],
        features = 512,
        name = "ECG_raw",
        dimension = 1,
        show_size = True,
        only_features_map = True,
    )

def create_model_fft():
    return CNN_model(
        input_shape = (None, 1),
        structures = [
            (32, 3, 0.0),
            (64, 3, 0.0),
            (128, 3, 0.0),
            (256, 3, 0.0),
            (512, 3, 0.0),
        ],
        features = 512,
        name = "ECG_fft",
        dimension = 1,
        show_size = True,
        only_features_map = True,
    )

def create_model():
    raw_model, _, _ = create_model_raw()
    fft_model, _, _ = create_model_fft()
    
    encoder = layers.concatenate([
        raw_model.output,
        fft_model.output,
    ])
    encoder = layers.Dense(1024, activation=layers.LeakyReLU(negative_slope=0.2))(encoder)
    

    if "show_size" in sys.argv:
        show_params(model, "ECG_combined")

    return model, encoder

save_path = path.join("res", "model_ECG.keras")
analyzer_path = path.join("res", "analyzer_ECG.keras")

if "epochs" in sys.argv:
    epochs = int(sys.argv[sys.argv.index("epochs")+1])
else:
    epochs = int(input("Please provide a valid number of epochs: "))
batch_size = 128

print("Creating model architecture...")
model, encoder = create_model()
analyzer = Model(inputs=model.input, outputs=encoder)

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
    start_from_epoch = 150,
)
cb_checkpoint = cbk.ModelCheckpoint(
    save_path, save_best_only = True
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
        hist = model.fit([X_raw_train, X_fft_train], 
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
    pred = model.predict([X_raw_test, X_fft_test], verbose=False)
    pred = [np.round(np.squeeze(x)) for x in pred]
    f = open(path.join("history", f"{id}_result_ECG.txt"), "w")
    print(classification_report(y_test, pred, target_names=["NO OSA", "OSA"]), file=f)
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm, file=f)
    names = ["loss"]
    names += [ f"threshold_0.{t}" for t in range(1, 10) ]
    results = model.evaluate([X_raw_test, X_fft_test], y_test, verbose=False)
    print("\nLoss and metrics", file=f)
    for idx in range(10):
        print(names[idx], ":", results[idx], file=f)
    f.close() 
    
    if "build" in sys.argv:
        for key, value in hist.history.items():
            data = np.array(value)
            his_path = path.join("history", f"{id}_{key}_ECG")
            np.save(his_path, data)
        print("Saving history done!")