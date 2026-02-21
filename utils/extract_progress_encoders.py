"""
ONE-TIME MIGRATION SCRIPT

PURPOSE:
  predict_userprogress.py needs LabelEncoders (Gender, Goal, Level, BMI_Category)
  and the feature order list to process user input before passing it to ONNX models.

  Currently these live inside model_progress.pickle alongside the XGBoost models.
  Since we no longer use the XGBoost models (replaced by ONNX), we extract ONLY
  the encoders and feature list into a small standalone file: progress_encoders.pickle

  After running this script:
  - model_progress.pickle  → No longer referenced in ANY code (stays on disk, unused)
  - progress_encoders.pickle → Loaded by predict_userprogress.py

EXPECTED OUTPUT:
  Loading model_progress.pickle...
  Encoders found: ['Gender', 'Goal', 'level', 'BMI_Category_x']
  Feature count: 18
  Saved: models/progress_encoders.pickle
  Done! You can now remove model_progress.pickle from active use.
"""

import os
import pickle

# Paths
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(BASE_DIR, "models", "model_progress.pickle")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "progress_encoders.pickle")

# Load source
print(f"Loading {SOURCE_PATH}...")

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(
        f"model_progress.pickle not found at:\n  {SOURCE_PATH}\n"
        "Make sure you run this script from the repo root."
    )

with open(SOURCE_PATH, "rb") as f:
    data_model = pickle.load(f)

# Extract only what we need
encoders = data_model["encoders"]
features = data_model["features"]

print(f"Encoders found: {list(encoders.keys())}")
print(f"Feature count: {len(features)}")
print(f"Feature order: {features}")

# Save slim artifact
slim = {
    "encoders": encoders,   # LabelEncoders for Gender, Goal, Level
    "features": features,   # Ordered list of 18 feature names
}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(slim, f)

print(f"\nSaved: {OUTPUT_PATH}")
print("Done! predict_userprogress.py will now load from progress_encoders.pickle")
print("model_progress.pickle is no longer referenced in any code.")