import pickle
import pandas as pd
import numpy as np
import os

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
    
except Exception as e:
    pass

def get_workout_plan_json(user_input: dict):
    try:
        goal_enc = encoders['goal'].transform([user_input.get('Goal', 'Weight Loss')])[0]
        level_enc = encoders['level'].transform([user_input.get('Level', 'Beginner')])[0]
        gender_enc = encoders['gender'].transform([user_input.get('Gender', 'Male')])[0]
        env_enc = encoders['environment'].transform([user_input.get('Environment', 'Gym')])[0]
    except Exception as e:
        return {"error": str(e), "status": "failed"}

    input_data = {
        'Goal_Encoded': goal_enc,
        'Workout_Frequency_x': user_input.get('Frequency', 3),
        'level_Encoded': level_enc,
        'Gender_Encoded': gender_enc,
        'Age_x': user_input.get('Age', 25),
        'Initial_Weight_kg_x': user_input.get('Weight', 60),
        'environment_Encoded': env_enc,
        'Badminton': int(user_input.get('Badminton', 0)),
        'Football': int(user_input.get('Football', 0)),
        'Basketball': int(user_input.get('Basketball', 0)),
        'Volleyball': int(user_input.get('Volleyball', 0)),
        'Swim': int(user_input.get('Swim', 0))
    }
    
    input_df = pd.DataFrame([input_data])[feature_order]
    
    try:
        input_scaled = scaler.transform(input_df)
        input_weighted = pd.DataFrame(input_scaled, columns=feature_order)
        
        for col, weight in weights.items():
            if col in input_weighted.columns:
                input_weighted[col] = input_weighted[col] * weight
                
        distances, indices = knn.kneighbors(input_weighted.values, n_neighbors=1)
        matched_user_id = db_profiles.iloc[indices[0][0]]['User_ID']
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

    weekly_plan = db_schedule.get(matched_user_id)
    
    if not weekly_plan:
        return {"error": "No schedule found for matched user.", "status": "failed"}

    return {
        "status": "success",
        "workout_plan": weekly_plan
    }