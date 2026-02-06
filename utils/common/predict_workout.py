import pickle
import pandas as pd
import numpy as np
import os
# ==========================================
# 1. LOAD THE OPTIMIZED MODEL (Global State)
# ==========================================
# Pastikan path ini benar relatif terhadap file main.py FastAPI kamu
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models', 'model_workout.pickle')

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    knn = model_data['knn_model']
    scaler = model_data['scaler']
    weights = model_data['weights']
    db_profiles = model_data['profiles_db']
    db_schedule = model_data['schedule_db']
    encoders = model_data['encoders']
    feature_order = model_data['features']
    
    print("✅ Workout Model Loaded Successfully.")
    
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    # Handle error gracefully di production

# ==========================================
# 2. PREDICTION SERVICE
# ==========================================

def get_workout_plan_json(user_input: dict):
    """
    Service logic untuk FastAPI. Menerima input user (dict),
    mencari jadwal yang cocok, dan mengembalikan JSON response.
    """
    
    # --- STEP 1: VALIDASI & ENCODING ---
    try:
        # Gunakan get dengan default value untuk menghindari KeyError
        # atau pastikan Pydantic Model di FastAPI sudah memfilter ini.
        goal_enc = encoders['goal'].transform([user_input.get('Goal', 'Weight Loss')])[0]
        level_enc = encoders['level'].transform([user_input.get('Level', 'Beginner')])[0]
        gender_enc = encoders['gender'].transform([user_input.get('Gender', 'Male')])[0]
        env_enc = encoders['environment'].transform([user_input.get('Environment', 'Gym')])[0]
        
    except ValueError as e:
        return {"error": f"Invalid category value. Details: {str(e)}", "status": "failed"}
    except Exception as e:
        return {"error": f"Encoding error: {str(e)}", "status": "failed"}

    # --- STEP 2: PREPARE FEATURE VECTOR ---
    input_data = {
        'Goal_Encoded': goal_enc,
        'Workout_Frequency_x': user_input.get('Frequency', 3),
        'level_Encoded': level_enc,
        'Gender_Encoded': gender_enc,
        'Age_x': user_input.get('Age', 25),
        'Initial_Weight_kg_x': user_input.get('Weight', 60),
        'environment_Encoded': env_enc,
        
        # Binary Flags (Pastikan input 0/1)
        'Badminton': int(user_input.get('Badminton', 0)),
        'Football': int(user_input.get('Football', 0)),
        'Basketball': int(user_input.get('Basketball', 0)),
        'Volleyball': int(user_input.get('Volleyball', 0)),
        'Swim': int(user_input.get('Swim', 0))
    }
    
    # Convert to DataFrame sesuai urutan fitur model
    input_df = pd.DataFrame([input_data])[feature_order]
    
    # --- STEP 3: SCALE & WEIGHT ---
    try:
        input_scaled = scaler.transform(input_df)
        input_weighted = pd.DataFrame(input_scaled, columns=feature_order)
        
        for col, weight in weights.items():
            if col in input_weighted.columns:
                input_weighted[col] = input_weighted[col] * weight
                
        # --- STEP 4: FIND NEAREST NEIGHBOR ---
        distances, indices = knn.kneighbors(input_weighted.values, n_neighbors=1)
        
        matched_index = indices[0][0]
        matched_user = db_profiles.iloc[matched_index]
        matched_user_id = matched_user['User_ID']
        
    except Exception as e:
        return {"error": f"Prediction engine error: {str(e)}", "status": "failed"}

    # --- STEP 5: RETRIEVE & FORMAT SCHEDULE ---
    schedule = db_schedule[db_schedule['User_ID'] == matched_user_id].copy()
    
    if schedule.empty:
        return {"error": "No schedule found for matched user.", "status": "failed"}

    # Cleaning: Replace NaN/Infinity with None (JSON compliant)
    schedule = schedule.where(pd.notnull(schedule), None)

    # Struktur JSON Respons
    # Kita akan mengelompokkan berdasarkan 'Day' agar lebih rapi di Frontend
    weekly_plan = {}
    
    # Sort dulu biar harinya urut (Day 1, Day 2...)
    # Asumsi kolom Day string "Day 1 - ...", kita bisa sort manual atau biarkan apa adanya
    # Kalau mau sort aman, bisa extract angka hari dulu, tapi ini opsional.
    
    unique_days = schedule['Day'].unique()
    
    for day in unique_days:
        day_exercises = schedule[schedule['Day'] == day]
        
        exercises_list = []
        for _, row in day_exercises.iterrows():
            exercises_list.append({
                "Muscle Group": row.get('Muscle Group'),
                "Exercise Name": row.get('Exercise Name'),
                "Sets": row.get('Sets'),
                "Reps": row.get('Reps'),
                "Calories_Burned": row.get('Calories_Burned'),
                "Duration_Minutes": row.get('Duration_Minutes'),
                "Rest_Minutes": row.get('Rest_Minutes'),
                "Equipment": row.get('Equipment'),
                "Instructions": row.get('Instructions')
            })
            
        weekly_plan[day] = exercises_list

    # --- FINAL RESPONSE ---
    response = {
        "status": "success",
        "matched_user_id": int(matched_user_id),
        "user_profile_match": {
            "goal": matched_user['Goal_x'],
            "frequency": int(matched_user['Workout_Frequency_x']),
            "level": matched_user.get('level_x', 'Unknown'),
            "environment": matched_user.get('Environment', 'Unknown')
        },
        "workout_plan": weekly_plan
    }
    
    return response