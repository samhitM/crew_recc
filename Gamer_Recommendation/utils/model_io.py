import os
from tensorflow.keras.models import load_model

def get_latest_model_path(directory='models/'):
    model_files = [f for f in os.listdir(directory) if f.endswith('.keras')]
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_model)

def load_latest_model(custom_objects=None):
    path = get_latest_model_path()
    if path:
        print(f"Loading model from: {path}")
        return load_model(path, custom_objects=custom_objects), path
    return None, None

def clean_old_models(directory='models/', keep_last_n=5):
    files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.keras')],
        key=os.path.getctime
    )
    for file in files[:-keep_last_n]:
        os.remove(file)
        print(f"Deleted old model: {file}")