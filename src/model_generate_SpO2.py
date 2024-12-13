import random
from model_functions import *

def create_model(name: str):
    inp = layers.Input(shape=(15,))
    shortcut = layers.Normalization()(inp)
    shortcut = layers.Dense(32)(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    shortcut = layers.Activation("relu")(shortcut)
    shortcut = layers.Dense(64)(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    shortcut = layers.Activation("relu")(shortcut)

    x = layers.Dense(128)(shortcut)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # residual connection
    x = layers.Add()([x, shortcut])
    
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    out = layers.Dense(15, activation="sigmoid")(x)
    
    model = Model(
        inputs = inp,
        outputs = out,
    )
    
    model.compile(
        optimizer = "adam",
        loss = "mse",
        metrics = ["mae"],
    )

    if "show_size" in sys.argv:
        show_params(model, name)
        
    return model

save_path = path.join("res", "model_generate_SpO2.keras")
model =  create_model("generate_SpO2")

if "epochs" in sys.argv:
    epochs = int(sys.argv[sys.argv.index("epochs")+1])
else:
    epochs = int(input("Please provide a valid number of epochs: "))
    
batch_size = 64
es_ep = 10
cb_early_stopping = cbk.EarlyStopping(
    restore_best_weights = True,
    start_from_epoch = es_ep,
)

if sys.argv[1] == "build":
    print("Loading data")
    X_pair = np.load(path.join("gen_data", "pair_ECG_SpO2.npy"))
    X, y = shuffle(X_pair[0], X_pair[1], random_state=np.random.randint(10000000))
    model.fit(
        X,
        y,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0.2,
        callbacks = [cb_early_stopping]
    )
    model.save(save_path)
    print("Done")