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
    
    if 'User_ID' in db_schedule.columns:
        db_schedule = db_schedule.set_index('User_ID')
        
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

    try:
        schedule = db_schedule.loc[[matched_user_id]].copy()
    except KeyError:
        return {"error": "No schedule found for matched user.", "status": "failed"}

    schedule = schedule.where(pd.notnull(schedule), None)
    
    rename_map = {
        'Muscle Group': 'muscle_group',
        'Exercise Name': 'exercise_name',
        'Sets': 'sets',
        'Reps': 'reps',
        'Calories_Burned': 'calories_burned',
        'Duration_Minutes': 'duration_minutes',
        'Rest_Minutes': 'rest_minutes',
        'Equipment': 'equipment',
        'Instructions': 'instructions'
    }
    schedule = schedule.rename(columns=rename_map)
    cols_to_keep = list(rename_map.values())
    
    weekly_plan = {}
    for day, group in schedule.groupby('Day'):
        weekly_plan[day] = group[cols_to_keep].to_dict('records')

    return {
        "status": "success",
        "workout_plan": weekly_plan
    }