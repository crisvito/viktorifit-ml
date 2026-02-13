import pickle
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. LOAD THE OPTIMIZED MODEL & DATABASE (Global State)
# ==============================================================================
# Objective: Load the pre-trained KNN model and reference databases once.
# 
# Why is this global?
# Loading large pickle files is expensive (slow). By loading it at the module level,
# we ensure this only happens once when the server starts, not every time a user 
# makes a request.

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models', 'model_workout.pickle')

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    knn = model_data['knn_model']      # The core ML algorithm
    scaler = model_data['scaler']      # For normalizing input data (0-1 range)
    weights = model_data['weights']    # Feature importance weights
    db_profiles = model_data['profiles_db'] # Reference user database
    db_schedule = model_data['schedule_db'] # Reference workout schedules
    encoders = model_data['encoders']  # LabelEncoders for categorical data
    feature_order = model_data['features']  # Exact column order for the model
    
    print("Workout Model Loaded Successfully.")
    
except Exception as e:
    print(f"Error Loading Model: {e}")
    # In a production environment, we might want to halt execution here
    # or trigger an alert because the API cannot function without the model.

# ==============================================================================
# 2. PREDICTION SERVICE
# ==============================================================================

def get_workout_plan_json(user_input: dict):
    """
    Generates a personalized workout schedule based on user characteristics.

    This service implements a 'Nearest Neighbor' approach. It takes the new user's
    attributes, finds the most similar user profile in our database (who has a 
    proven successful routine), and returns that existing user's workout plan.

    ---------------------------------------------------------------------------
    Args:
        user_input (dict): A dictionary containing user stats 
                           (Age, Goal, Level, Environment, etc.).

    Returns:
        dict: A JSON-compatible dictionary containing the status and 
              the full weekly workout schedule grouped by day.
    """
    
    # --- STEP 1: VALIDATION & ENCODING ---
    # Objective: Convert text inputs (e.g., 'Male', 'Weight Loss') into numbers.
    # Why? Machine learning models can only understand numerical vectors.
    try:
        # Use .get() with defaults to prevent crashes if keys are missing.
        # Note: The FastAPI Pydantic model should technically catch missing fields,
        # but this acts as a second layer of defense.
        goal_enc = encoders['goal'].transform([user_input.get('Goal', 'Weight Loss')])[0]
        level_enc = encoders['level'].transform([user_input.get('Level', 'Beginner')])[0]
        gender_enc = encoders['gender'].transform([user_input.get('Gender', 'Male')])[0]
        env_enc = encoders['environment'].transform([user_input.get('Environment', 'Gym')])[0]
        
    except ValueError as e:
        # Handle cases where the user sends a category not in our encoders
        return {"error": f"Invalid category value. Details: {str(e)}", "status": "failed"}
    except Exception as e:
        return {"error": f"Encoding error: {str(e)}", "status": "failed"}

    # --- STEP 2: PREPARE FEATURE VECTOR ---
    # Objective: Construct the data row exactly as the model expects it.
    input_data = {
        'Goal_Encoded': goal_enc,
        'Workout_Frequency_x': user_input.get('Frequency', 3),
        'level_Encoded': level_enc,
        'Gender_Encoded': gender_enc,
        'Age_x': user_input.get('Age', 25),
        'Initial_Weight_kg_x': user_input.get('Weight', 60),
        'environment_Encoded': env_enc,
        
        # Binary Flags (0 or 1) for Hobbies/Sports
        # These help refine the neighbor search by matching activity patterns.
        'Badminton': int(user_input.get('Badminton', 0)),
        'Football': int(user_input.get('Football', 0)),
        'Basketball': int(user_input.get('Basketball', 0)),
        'Volleyball': int(user_input.get('Volleyball', 0)),
        'Swim': int(user_input.get('Swim', 0))
    }
    
    # Create DataFrame and enforce column order
    # Why? Scikit-learn models are strict about column order. 
    # If we swap 'Age' and 'Weight', the prediction will be wrong.
    input_df = pd.DataFrame([input_data])[feature_order]
    
    # --- STEP 3: SCALE & WEIGHT ---
    # Objective: Normalize data and apply custom importance weights.
    try:
        # Normalize features to 0-1 range to prevent large numbers (like Weight)
        # from dominating small numbers (like Frequency).
        input_scaled = scaler.transform(input_df)
        input_weighted = pd.DataFrame(input_scaled, columns=feature_order)
        
        # Apply Weights:
        # We manually increase the value of certain columns (defined in training)
        # to tell the KNN that matching 'Goal' is more important than matching 'Age'.
        for col, weight in weights.items():
            if col in input_weighted.columns:
                input_weighted[col] = input_weighted[col] * weight
                
        # --- STEP 4: FIND NEAREST NEIGHBOR (KNN) ---
        # Query the model to find the single most similar profile (k=1)
        distances, indices = knn.kneighbors(input_weighted.values, n_neighbors=1)
        
        # Retrieve the User ID of the matched profile
        matched_index = indices[0][0]
        matched_user = db_profiles.iloc[matched_index]
        matched_user_id = matched_user['User_ID']
        
    except Exception as e:
        return {"error": f"Prediction engine error: {str(e)}", "status": "failed"}

    # --- STEP 5: RETRIEVE & FORMAT SCHEDULE ---
    # Objective: Fetch the workout routine associated with the matched User ID.
    schedule = db_schedule[db_schedule['User_ID'] == matched_user_id].copy()
    
    if schedule.empty:
        return {"error": "No schedule found for matched user.", "status": "failed"}

    # Cleaning: JSON cannot handle NaN or Infinity values, so we replace them with None.
    schedule = schedule.where(pd.notnull(schedule), None)

    # Structure the JSON Response
    # We group exercises by 'Day' to make it easy for the Frontend to display.
    weekly_plan = {}
    
    # Identify unique workout days in the schedule
    unique_days = schedule['Day'].unique()
    
    for day in unique_days:
        # Filter exercises for the specific day
        day_exercises = schedule[schedule['Day'] == day]
        
        exercises_list = []
        for _, row in day_exercises.iterrows():
            # Build the exercise object
            exercises_list.append({
                "muscle_group": row.get('Muscle Group'),
                "exercise_name": row.get('Exercise Name'),
                "sets": row.get('Sets'),
                "reps": row.get('Reps'),
                "calories_burned": row.get('Calories_Burned'),
                "duration_minutes": row.get('Duration_Minutes'),
                "rest_minutes": row.get('Rest_Minutes'),
                "equipment": row.get('Equipment'),
                "instructions": row.get('Instructions')
            })
            
        weekly_plan[day] = exercises_list

    # --- FINAL RESPONSE ---
    response = {
        "status": "success",
        "workout_plan": weekly_plan
    }
    
    return response