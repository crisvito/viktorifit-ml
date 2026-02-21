from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

from utils.common.predict_workout import get_workout_plan_json
from utils.common.predict_meal import get_meal_plan_json
from utils.common.predict_userprogress import get_progress_roadmap

# ==============================================================================
# ENVIRONMENT SETUP & PATH CONFIGURATION
# ==============================================================================
# Objective: Enable absolute imports from parent directories.
# Why is this necessary?
# Python by default only looks in the current directory. Since our utility scripts
# are located in a parent folder structure, we append the parent directory to 
# sys.path to prevent "ModuleNotFoundError".
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


# ==============================================================================
# DATA VALIDATION SCHEMAS (PYDANTIC MODELS)
# ==============================================================================
# Objective: Enforce strict data typing and validation for incoming API requests.
# Why is this necessary?
# These classes act as a "Contract" between the Client (Frontend) and the Server.
# They ensure that data like 'Age' is always an integer and 'Weight' is a float
# before any logic is executed, preventing runtime type errors.

class WorkoutRequest(BaseModel):
    """
    Schema representing the User Profile & Preferences for Workout Generation.

    This model captures all physiological data and user constraints required
    to build a personalized workout routine.

    Attributes:
        Age (int): User's age in years.
        Gender (str): User's biological sex (e.g., 'Male', 'Female').
        Height_cm (float): Height in centimeters.
        Weight_kg (float): Weight in kilograms.
        Body_Fat_Category (int): Categorical index of body fat (e.g., 1=Low, 2=Medium, etc.).
        Body_Fat_Percentage (float): Precise body fat percentage value.
        Goal (str): The fitness objective (e.g., 'Bulking', 'Cutting').
        Frequency (int): Number of workout days per week.
        Duration (int): Duration of each session in minutes.
        Level (str): User experience level (e.g., 'Beginner', 'Advanced').
        Environment (str): Available facility ('Gym' or 'Home').
        
        Hobbies (Binary Flags - 0 or 1):
        These fields indicate if a user plays specific sports. 
        Used to adjust cardio recommendations.
        - Badminton, Football, Basketball, Volleyball, Swim.
    """
    Age: int
    Gender: str
    Height_cm: float
    Weight_kg: float
    Body_Fat_Category: int
    Body_Fat_Percentage: float

    Goal: str
    Frequency: int
    Duration: int
    Level: str
    Environment: str

    Badminton: int = 0
    Football: int = 0
    Basketball: int = 0
    Volleyball: int = 0
    Swim: int = 0

class MealRequest(BaseModel):
    """
    Schema representing Nutritional Targets for Meal Planning.

    This model serves as the input for the diet recommendation engine, 
    focusing purely on macronutrient distribution.

    Attributes:
        Daily_Calories (float): Total energy intake target for the day.
        Target_Protein_g (float): Target protein mass in grams.
        Target_Carbs_g (float): Target carbohydrate mass in grams.
        Target_Fat_g (float): Target fat mass in grams.
        Frequency (int): Number of meals to split the macros into (Default: 3).
    """
    Daily_Calories: float 
    Target_Protein_g: float
    Target_Carbs_g: float
    Frequency: int = 3

class ProgressRequest(BaseModel):
    """
    Schema representing the Baseline for Progress Forecasting.

    This model captures the user's starting point (T0) to calculate 
    a 12-week roadmap of expected physiological changes.

    Attributes:
        Age (int): User's age.
        Gender (str): User's gender used for metabolic rate calculation.
        Height_cm (float): User's height.
        Initial_Weight_kg (float): Starting weight at Week 0.
        Goal (str): The target outcome (influences the curve of the graph).
        Level (str): Fitness experience (influences adaptation rate).
        Body_Fat_Category (int): Initial body fat category.
        Body_Fat_Percentage (float): Initial body fat percentage.
        Frequency (int): Workout frequency per week.
        Duration (int): Workout duration per session.
        
        Hobbies (Binary Flags):
        Sports activities that contribute to additional caloric burn (TDEE).
    """
    Age: int
    Gender: str
    Height_cm: float
    Initial_Weight_kg: float
    Goal: str
    Level: str
    
    Body_Fat_Category: int
    
    Frequency: int
    Duration: int
    
    # Hobi (Opsional)
    Badminton: int = 0
    Football: int = 0
    Basketball: int = 0
    Volleyball: int = 0
    Swim: int = 0


# ==============================================================================
# API INITIALIZATION & ENDPOINTS
# ==============================================================================

app = FastAPI()

@app.get("/")
def home():
    """
    Health Check / Root Endpoint.
    
    Returns:
        dict: A simple status message confirming the API is active.
    """
    return {"message": "Workout AI API is Running!"}

@app.post("/workout")
def predict_workout_endpoint(user_data: WorkoutRequest):
    """
    Orchestrates the Workout Plan Generation Process.

    This endpoint acts as a controller that receives raw user data, 
    validates it against the schema, and delegates the logic to the 
    workout prediction engine.

    ---------------------------------------------------------------------------
    Args:
        user_data (WorkoutRequest): The validated JSON body containing 
                                    user stats and preferences.
    
    Returns:
        dict: A complex JSON object containing the full workout schedule.
    
    Raises:
        HTTPException (400): If the prediction engine returns a 'failed' status 
                            (e.g., due to internal calculation errors).
    """
    # 1. Data Transformation (Model -> Dict)
    # Convert Pydantic model to a standard dictionary for internal processing.
    input_dict = user_data.model_dump() 
    
    # 2. Service Invocation (The "Brain")
    # Pass the data to the external utility function that contains the ML/Logic.
    result = get_workout_plan_json(input_dict)
    
    # 3. Error Handling & Response
    # Check if the utility function flagged an error during processing.
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error"))
        
    return result

@app.post("/meal")
def predict_meal_endpoint(meal_data: MealRequest):
    """
    Orchestrates the Meal Plan Generation Process.

    Takes calculated macronutrient targets and matches them with appropriate
    food items from the database to create a daily menu.

    ---------------------------------------------------------------------------
    Args:
        meal_data (MealRequest): The validated JSON body containing 
                                calorie and macro targets.
    
    Returns:
        dict: A JSON object structured by meal time (Breakfast, Lunch, Dinner).
        
    Raises:
        HTTPException (400): If the meal allocation logic fails.
    """
    # 1. Data Transformation
    input_dict = meal_data.model_dump()
    
    # 2. Service Invocation
    # Calls the algorithm that knapsacks/optimizes food choices based on macros.
    result = get_meal_plan_json(input_dict)
    
    # 3. Error Handling
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error"))
        
    return result

@app.post("/userprogress")
def predict_progress_endpoint(data: ProgressRequest):
    """
    Orchestrates the 12-Week Progress Roadmap Calculation.

    This endpoint triggers a simulation loop that estimates body composition
    changes week-by-week based on the user's starting point and activity level.

    ---------------------------------------------------------------------------
    Args:
        data (ProgressRequest): The validated JSON body containing 
                                baseline stats and activity variables.
    
    Returns:
        dict: A JSON wrapper containing a list of weekly progress objects.
            Example: { "roadmap": [ {Week 1...}, {Week 2...} ] }
    
    Raises:
        HTTPException (400): If the forecasting simulation fails.
    """
    input_dict = data.model_dump()
    
    # Panggil fungsi looping baru kita
    # (Triggers the weekly simulation logic)
    result = get_progress_roadmap(input_dict)
    
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error"))
        
    return result