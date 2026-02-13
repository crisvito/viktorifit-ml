import os
import numpy as np
import pickle
import onnxruntime as ort

# ==============================================================================
# 1. ENVIRONMENT SETUP & RESOURCE LOADING
# ==============================================================================
# Objective: Initialize paths and load static assets required for inference.
# 
# - BASE_DIR: Ensures the script can find resources regardless of where it's run.
# - PICKLE_PATH: Contains the 'Encoders' (translators from Text -> Number) and 
#   'Feature Names' (the exact order of inputs required by the AI).

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
PICKLE_PATH = os.path.join(MODEL_DIR, "model_progress.pickle") 

# Load the Data Artifacts
# We need the exact encoders used during training to ensure consistency.
with open(PICKLE_PATH, "rb") as f:
    data_model = pickle.load(f)

progress_encoders = data_model["encoders"]
# CRITICAL: The order of features in this list must match the training phase 100%.
progress_features = data_model["features"] 

# ==============================================================================
# 2. ONNX RUNTIME OPTIMIZATION & LOADING
# ==============================================================================
# Objective: Load AI models into memory with performance optimizations.
#
# Optimization Strategy:
# We set 'intra_op_num_threads' to 1. Since our models are small and we might 
# process many requests, avoiding multi-threading overhead per model actually 
# makes individual predictions faster (lower latency).

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_sessions = {}
model_files = [f for f in os.listdir(ONNX_DIR) if f.endswith(".onnx")]

print(f"Loading {len(model_files)} ONNX models...")

# Load each .onnx file into a session
for file in model_files:
    name = file.replace(".onnx", "")
    path = os.path.join(ONNX_DIR, file)
    # CPUExecutionProvider is sufficient and stable for this workload
    onnx_sessions[name] = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

# --- WARM-UP ROUTINE ---
# Why is this necessary?
# The first time an ONNX session runs, it performs memory allocation and 
# graph initialization, causing a delay (Cold Start). 
# We run a dummy inference now so the first real user request is instant.
dummy_input = np.zeros((1, len(progress_features)), dtype=np.float32)
for name, sess in onnx_sessions.items():
    input_name = sess.get_inputs()[0].name
    sess.run(None, {input_name: dummy_input})

print("Models loaded & warmed up!")

# ==============================================================================
# 3. FAST PREDICTION ENGINE (SINGLE WEEK)
# ==============================================================================

def predict_single_week_fast(user_input: dict, week: int, current_weight: float):
    """
    Performs inference for a specific week based on the user's current state.

    This function prepares the data vector, encodes categorical values, 
    and queries all loaded ONNX models to predict physical and nutritional 
    metrics for the given week.

    ---------------------------------------------------------------------------
    Args:
        user_input (dict): The static user profile (Age, Height, Gender, etc.).
        week (int): The specific week number (1-12) to predict for.
        current_weight (float): The user's weight *at the start of this week*.
                                (This changes dynamically in the roadmap loop).

    Returns:
        dict: A dictionary containing predicted values (e.g., {'Weight_kg': 70.5, ...}).
    """
    # --- A. Preprocessing (Optimized for Speed) ---
    # Convert height to meters for BMI calculation
    tinggi_m = user_input["Height_cm"] / 100
    
    # Safety Guard: Prevent DivisionByZero error if height is invalid
    bmi = round(current_weight / (tinggi_m ** 2), 2) if tinggi_m > 0 else 0

    # Determine BMI Category logic
    if bmi < 18.5: cat_bmi = "Underweight"
    elif bmi < 25: cat_bmi = "Normal"
    elif bmi < 30: cat_bmi = "Overweight"
    else: cat_bmi = "Obese"

    # Encoding: Transform categorical strings into numerical vectors.
    # Uses try-except blocks as a fallback mechanism to prevent crashes 
    # if an unknown category is encountered.
    try:
        gender_code = progress_encoders["Gender"].transform([user_input["Gender"]])[0]
        goal_code = progress_encoders["Goal"].transform([user_input["Goal"]])[0]
        level_code = progress_encoders["level"].transform([user_input["Level"]])[0]
        bmi_cat_code = progress_encoders["BMI_Category_x"].transform([cat_bmi])[0]
    except:
        # Fallback: Assign default index (0) if encoding fails
        gender_code = 0; goal_code = 0; level_code = 0; bmi_cat_code = 0

    # Data Construction: Build the input vector.
    # CRITICAL: This list matches the 'progress_features' order exactly.
    input_values = [
        user_input["Age"],
        gender_code,
        user_input["Height_cm"],
        current_weight,           # Dynamic variable
        bmi,                      # Dynamic variable
        bmi_cat_code,             # Dynamic variable
        user_input.get("Body_Fat_Category", 0),
        user_input["Body_Fat_Percentage"],
        goal_code,
        user_input["Frequency"],
        user_input["Duration"],
        level_code,
        # Sports/Hobbies (Binary Flags)
        user_input.get("Badminton", 0),
        user_input.get("Football", 0),
        user_input.get("Basketball", 0),
        user_input.get("Volleyball", 0),
        user_input.get("Swim", 0),
        week                      # Time variable
    ]
    
    # Convert to Float32 NumPy array (Required by ONNX)
    # Shape: (1 sample, N features)
    input_array = np.array([input_values], dtype=np.float32)

    # --- B. Inference Loop ---
    hasil = {}
    
    # Iterate through all loaded models (Weight, BodyFat, Calories, etc.)
    # Since operations are optimized in C++, this loop is extremely fast.
    for name, session in onnx_sessions.items():
        input_name = session.get_inputs()[0].name
        
        # Execute Prediction
        pred_raw = session.run(None, {input_name: input_array})[0][0]
        
        # --- C. Post-Processing ---
        # Decode output if the model predicts a category (indicated by "Encoded" in filename)
        if "Encoded" in name:
            col = name.replace("_Encoded", "")
            val_int = int(round(pred_raw))
            
            # Inverse Transform: Number -> String
            try:
                # Assuming the column name matches the encoder key
                enc_name = col
                decoded = progress_encoders[enc_name].inverse_transform([val_int])[0]
                hasil[col] = decoded
            except:
                hasil[col] = val_int # Return raw number if decoding fails
        else:
            # Handle Numeric Outputs
            # Use Integers for Calories/mg/ml, Floats for others
            if any(x in name for x in ["Calories", "mg", "ml"]):
                hasil[name] = int(pred_raw)
            else:
                hasil[name] = round(float(pred_raw), 2)

    return hasil

# ==============================================================================
# 4. ROADMAP GENERATION (ORCHESTRATOR)
# ==============================================================================

def get_progress_roadmap(user_input: dict):
    """
    Generates a 12-week physiological progression roadmap.

    This function simulates the user's journey by chaining predictions:
    the predicted weight of Week 1 becomes the input weight for Week 2.
    This creates a realistic curve of progress rather than static estimates.

    ---------------------------------------------------------------------------
    Args:
        user_input (dict): The initial user data (T=0).

    Returns:
        dict: A structured response containing metadata and the list of 
                weekly projections.
    """
    roadmap = []
    current_weight = user_input["Initial_Weight_kg"]

    # Simulation Loop: Iterate from Week 1 to Week 12
    for week in range(1, 13): 
        # 1. Predict stats for this specific week
        pred = predict_single_week_fast(user_input, week, current_weight)

        # 2. State Update (The Chaining Effect)
        # Update 'current_weight' so the next iteration uses the new value.
        if "Weight_kg" in pred:
            current_weight = pred["Weight_kg"]

        # 3. Result Formatting
        # Structure the data into a clean JSON format for the Frontend
        roadmap.append({
            "week": week,
            "physical": {
                "weight_kg": pred.get("Weight_kg"),
                # "bmi": pred.get("BMI"), 
                # "bmi_category": pred.get("BMI_Category_y"),
                "body_fat_percentage": pred.get("Body_Fat_Percentage_y"),
            },
            "nutrition": {
                "calories": pred.get("Daily_Calories"),
                "water_ml": pred.get("Daily_Water_ml"),
                "sugar_limit_g": pred.get("Limit_Sugar_g"),
                # "cholesterol_limit_mg": pred.get("Limit_Cholesterol_mg")
            },
            "macro": {
                "protein_g": pred.get("Target_Protein_g"),
                "carbs_g": pred.get("Target_Carbs_g"),
                "fat_g": pred.get("Target_Fat_g"),
                "fiber_g": pred.get("Target_Fiber_g"),
                # "calcium_mg": pred.get("Target_Calcium_mg")
            }
        })

    return {
        "status": "success",
        "total_weeks": 12,
        "roadmap": roadmap
    }