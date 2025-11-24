import os
import pandas as pd
import joblib
from .encode_utils import encode_value

# Folder data
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
SCALER_PATH = os.path.join(DATA_PATH, "scaler")

SCALER_FILE = os.path.join(SCALER_PATH, "scaler.pkl")
scaler = joblib.load(SCALER_FILE)

SCALER_FEATURES_FILE = os.path.join(SCALER_PATH, "scaler_features.pkl")
scaler_features = joblib.load(SCALER_FEATURES_FILE)

model_names = [
    "Exercises_encoded",
    "Equipment_encoded",
    "Diet_encoded",
    "Recommendation_encoded"
]

models = {}
for name in model_names:
    file_path = os.path.join(MODELS_PATH, f"{name}_Random_Forest.pkl")
    if os.path.exists(file_path):
        models[name] = joblib.load(file_path)
    else:
        raise FileNotFoundError(f"File model tidak ditemukan: {file_path}")
    
def predict_user_input(user_input):
    input_values = {}
    for col, val in user_input.items():
        if isinstance(val, str) and col.endswith('_encoded'):
            val = encode_value(col, val)
        input_values[col] = val

    df_scaled = pd.DataFrame([input_values])
    df_scaled[scaler_features] = scaler.transform(df_scaled[scaler_features])

    predictions = {}
    for col, model in models.items():
        predictions[col] = model.predict(df_scaled)[0]

    return predictions

print("Predict utils loaded successfully.")
