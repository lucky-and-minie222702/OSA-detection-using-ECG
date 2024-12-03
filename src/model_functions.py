from timeit import default_timer as timer
import keras

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

cb = TimingCallback()

def convert_bytes(byte_size):
    units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]
    size = byte_size
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

def convert_minutes(total_minutes):
    total_seconds = total_minutes * 60
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def show_params(model, name):
    print(f"Model {name}:")
    params = model.count_params()
    print(" | Total params:", "{:,}".format(params).replace(",", " "))
    print(" | Size        :", convert_bytes(params * 4))