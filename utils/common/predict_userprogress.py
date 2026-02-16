"""
utils/common/predict_userprogress.py
--------------------------------------
Generates a 12-week physiological progression roadmap using ONNX inference.

MODEL ARCHITECTURE:
  - Encoders + feature order  → models/progress_encoders.pickle  (small, ~10 KB)
  - Progress predictions      → models/onnx/*.onnx               (9 XGBoost models)

NOTE: model_progress.pickle is FULLY DEPRECATED.
      Run scripts/extract_progress_encoders.py once to produce progress_encoders.pickle.
"""

import os
import numpy as np
import pickle
import onnxruntime as ort

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================

# Walk 3 levels up from this file's location to reach the repo root.
# utils/common/predict_userprogress.py -> utils/common -> utils -> repo root
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
ONNX_DIR    = os.path.join(MODEL_DIR, "onnx")

# Slim artifact: only encoders + feature order (no XGBoost objects)
ENCODERS_PATH = os.path.join(MODEL_DIR, "progress_encoders.pickle")

# ==============================================================================
# 2. LOAD ENCODERS & FEATURE ORDER
# ==============================================================================
# We load a lightweight pickle that contains ONLY:
#   - LabelEncoders: Gender, Goal, level, BMI_Category_x
#   - features: ordered list of 18 column names
#
# This replaces the old model_progress.pickle dependency.

print("Loading progress encoders...")
try:
    with open(ENCODERS_PATH, "rb") as f:
        encoders_data = pickle.load(f)

    progress_encoders = encoders_data["encoders"]
    progress_features = encoders_data["features"]

    print(f"  Encoders loaded: {list(progress_encoders.keys())}")
    print(f"  Feature count  : {len(progress_features)}")

except FileNotFoundError:
    raise FileNotFoundError(
        f"\n[predict_userprogress] progress_encoders.pickle not found at:\n"
        f"  {ENCODERS_PATH}\n\n"
        "Run this once from the repo root to generate it:\n"
        "  python scripts/extract_progress_encoders.py\n"
    )
except Exception as e:
    raise RuntimeError(f"[predict_userprogress] Failed to load encoders: {e}")

# ==============================================================================
# 3. LOAD & WARM UP ONNX SESSIONS
# ==============================================================================
# Strategy: Load all 9 .onnx files into InferenceSessions with optimizations,
# then do a dummy run to eliminate cold-start latency on the first real request.

so = ort.SessionOptions()
so.intra_op_num_threads = 1         # Avoid threading overhead for small models
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_sessions = {}

if not os.path.isdir(ONNX_DIR):
    raise FileNotFoundError(
        f"[predict_userprogress] ONNX directory not found: {ONNX_DIR}"
    )

model_files = [f for f in os.listdir(ONNX_DIR) if f.endswith(".onnx")]

if not model_files:
    raise FileNotFoundError(
        f"[predict_userprogress] No .onnx files found in: {ONNX_DIR}"
    )

print(f"Loading {len(model_files)} ONNX models from {ONNX_DIR}...")

for file in model_files:
    name = file.replace(".onnx", "")
    path = os.path.join(ONNX_DIR, file)
    onnx_sessions[name] = ort.InferenceSession(
        path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    print(f"  ✅ {name}")

# ─── Warm-up: run a dummy inference on all sessions ──────────────────────────
# This triggers ONNX memory allocation now so the first real request is instant.
dummy_input = np.zeros((1, len(progress_features)), dtype=np.float32)

for name, sess in onnx_sessions.items():
    inp_name = sess.get_inputs()[0].name
    sess.run(None, {inp_name: dummy_input})

print(f"All {len(onnx_sessions)} ONNX models loaded & warmed up!")


# ==============================================================================
# 4. BMI HELPERS
# ==============================================================================

def _compute_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    if height_m <= 0:
        return 0.0
    return round(weight_kg / (height_m ** 2), 2)


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:  return "Underweight"
    if bmi < 25.0:  return "Normal"
    if bmi < 30.0:  return "Overweight"
    return "Obese"


# ==============================================================================
# 5. SINGLE-WEEK PREDICTION ENGINE
# ==============================================================================

def _predict_single_week(user_input: dict, week: int, current_weight: float) -> dict:
    """
    Run all ONNX models for one specific week.

    Args:
        user_input    : Static user profile dict (Age, Height_cm, Goal, etc.)
        week          : Week number 1–12.
        current_weight: Dynamic weight at the START of this week.
                        (Updates each iteration in the roadmap loop.)

    Returns:
        dict: Raw predictions keyed by model name, e.g. {"Weight_kg": 68.2, ...}
    """

    # ─── A. Derived features ─────────────────────────────────────────────────
    bmi = _compute_bmi(current_weight, user_input["Height_cm"])
    cat_bmi = _bmi_category(bmi)

    # ─── B. Encode categorical inputs ────────────────────────────────────────
    # Use try/except so an unknown category doesn't crash the whole roadmap.
    try:
        gender_code  = progress_encoders["Gender"].transform([user_input["Gender"]])[0]
        goal_code    = progress_encoders["Goal"].transform([user_input["Goal"]])[0]
        level_code   = progress_encoders["level"].transform([user_input["Level"]])[0]
        bmi_cat_code = progress_encoders["BMI_Category_x"].transform([cat_bmi])[0]
    except Exception:
        gender_code = goal_code = level_code = bmi_cat_code = 0

    # ─── C. Build input vector (must match progress_features order exactly) ──
    input_values = [
        user_input["Age"],
        gender_code,
        user_input["Height_cm"],
        current_weight,                          # dynamic
        bmi,                                     # dynamic (derived)
        bmi_cat_code,                            # dynamic (derived)
        user_input.get("Body_Fat_Category", 0),
        user_input["Body_Fat_Percentage"],
        goal_code,
        user_input["Frequency"],
        user_input["Duration"],
        level_code,
        user_input.get("Badminton",   0),
        user_input.get("Football",    0),
        user_input.get("Basketball",  0),
        user_input.get("Volleyball",  0),
        user_input.get("Swim",        0),
        week,                                    # time variable
    ]

    input_array = np.array([input_values], dtype=np.float32)

    # ─── D. Run all ONNX sessions ─────────────────────────────────────────────
    results = {}

    for name, session in onnx_sessions.items():
        inp_name = session.get_inputs()[0].name

        try:
            raw = session.run(None, {inp_name: input_array})[0][0]
        except Exception as e:
            print(f"  [WARN] ONNX [{name}] inference error: {e}")
            results[name] = None
            continue

        # ─── E. Post-process output ───────────────────────────────────────────
        # Models whose filenames contain "Encoded" output a category index.
        # All others output a numeric value directly.
        if "Encoded" in name:
            col = name.replace("_Encoded", "")
            idx = int(round(float(raw)))
            try:
                results[col] = progress_encoders[col].inverse_transform([idx])[0]
            except Exception:
                results[col] = idx
        else:
            # Use int for large whole-number metrics, float for everything else
            if any(x in name for x in ["Calories", "_mg", "_ml"]):
                results[name] = int(raw)
            else:
                results[name] = round(float(raw), 2)

    return results


# ==============================================================================
# 6. ROADMAP GENERATOR  (called by app.py)
# ==============================================================================

def get_progress_roadmap(user_input: dict) -> dict:
    """
    Simulate a 12-week fitness progression roadmap.

    Uses a chaining strategy: the predicted Weight_kg at Week N
    becomes the input weight for Week N+1, producing a realistic curve.

    Args:
        user_input: Validated user dict from ProgressRequest (via app.py).
                    Required keys:
                      Age, Gender, Height_cm, Initial_Weight_kg,
                      Goal, Level, Body_Fat_Category, Body_Fat_Percentage,
                      Frequency, Duration
                    Optional (default 0):
                      Badminton, Football, Basketball, Volleyball, Swim

    Returns:
        dict: {
            "status": "success",
            "total_weeks": 12,
            "roadmap": [ { week, physical, nutrition, macro }, ... ]
        }
    """
    try:
        roadmap = []
        current_weight = user_input["Initial_Weight_kg"]

        for week in range(1, 13):
            pred = _predict_single_week(user_input, week, current_weight)

            # Chain: update weight for next iteration
            if pred.get("Weight_kg") is not None:
                current_weight = pred["Weight_kg"]

            roadmap.append({
                "week": week,
                "physical": {
                    "weight_kg":           pred.get("Weight_kg"),
                    "body_fat_percentage": pred.get("Body_Fat_Percentage_y"),
                },
                "nutrition": {
                    "calories":      pred.get("Daily_Calories"),
                    "water_ml":      pred.get("Daily_Water_ml"),
                    "sugar_limit_g": pred.get("Limit_Sugar_g"),
                },
                "macro": {
                    "protein_g": pred.get("Target_Protein_g"),
                    "carbs_g":   pred.get("Target_Carbs_g"),
                    "fat_g":     pred.get("Target_Fat_g"),
                    "fiber_g":   pred.get("Target_Fiber_g"),
                },
            })

        return {
            "status":      "success",
            "total_weeks": 12,
            "roadmap":     roadmap,
        }

    except Exception as e:
        return {
            "status": "failed",
            "error":  str(e),
        }