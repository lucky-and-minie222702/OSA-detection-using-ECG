from keras.saving import load_model
from os import listdir
from os.path import isfile, join
from pathlib import Path

model_path = "res"
model_files = [join(model_path, f) for f in listdir(model_path) if isfile(join(model_path, f))]

for model_name in model_files:
    print(model_name)
    model = load_model(model_name)
    raw_name = Path(model_name).stem
    model.export(join("exported", raw_name))
    model.save(join("exported", raw_name+".h5"), save_format='h5')

print("Done!")