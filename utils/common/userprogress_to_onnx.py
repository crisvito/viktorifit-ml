import pickle
import os
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx

# ==============================================================================
# MODEL CONVERSION UTILITY (Pickle -> ONNX)
# ==============================================================================
# Objective: Convert heavy Python-based XGBoost models into optimized ONNX format.
#
# Why is this necessary?
# 1. Performance: ONNX Runtime is significantly faster (C++ based) than native Python inference.
# 2. Portability: ONNX models are language-agnostic and easier to deploy.
# 3. Memory: Loading individual .onnx files is more memory-efficient than loading 
#    one giant .pickle file containing all model objects.

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# Define directory structures relative to the script location to ensure 
# the script runs correctly on any developer machine or server.

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(current_dir))

# Input: The large dictionary containing all trained XGBoost models
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_progress.pickle")

# Output: Directory where optimized .onnx files will be saved
ONNX_DIR = os.path.join(BASE_DIR, "models", "onnx")

# Create output directory if it doesn't exist to prevent FileNotFoundError
os.makedirs(ONNX_DIR, exist_ok=True)

# ==============================================================================
# 2. LOAD TRAINED ARTIFACTS
# ==============================================================================
print("Loading pickle model...")

# Load the pickle file to access the raw XGBoost objects
with open(MODEL_PATH, "rb") as f:
    data_model = pickle.load(f)

# Extract the dictionary of models and the feature list used during training
progress_models = data_model["models_dict"]
features = data_model["features"]
n_features = len(features)

print(f"Total features: {n_features}")
print("Converting models to ONNX...")

# ==============================================================================
# 3. CONVERSION LOOP
# ==============================================================================
# Iterate through every target variable (e.g., Weight, BodyFat, Calories) 
# and convert its specific model to ONNX.

for target_name, info in progress_models.items():
    model = info["model"]
    
    # --- CRITICAL STEP: REMOVE FEATURE NAMES ---
    # Why? 
    # XGBoost models saved with specific feature names often cause errors during 
    # ONNX conversion if the input tensor naming doesn't match perfectly.
    # By setting names to None, we force the model to accept inputs based on 
    # COLUMN ORDER (Position) rather than COLUMN NAME. This is safer for deployment.
    booster = model.get_booster()
    booster.feature_names = None

    # Define Input Schema
    # [None, n_features] means:
    # - None: Accepts any batch size (1 row or 100 rows).
    # - n_features: Must match the exact number of input columns.
    initial_type = [
        ('float_input', FloatTensorType([None, n_features]))
    ]
    
    # Convert using ONNXMLTools (Specialized for XGBoost/Scikit-Learn)
    onnx_model = onnxmltools.convert_xgboost(
        model,
        initial_types=initial_type
    )
    
    # Save the optimized model to disk
    save_path = os.path.join(ONNX_DIR, f"{target_name}.onnx")
    onnx.save_model(onnx_model, save_path)
    
    print(f"Saved: {target_name}.onnx")

print("All models successfully converted to ONNX!")