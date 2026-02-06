from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import sys
import os

# --- TRIK IMPORT FOLDER TETANGGA ---
# Ambil path folder saat ini (app), lalu mundur satu langkah ke (project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Sekarang import ini PASTI BERHASIL
from utils.common.predict_workout import get_workout_plan_json
from utils.common.predict_meal import get_meal_plan_json
from utils.common.predict_userprogress import get_progress_prediction
from utils.common.predict_userprogress import get_progress_roadmap


class WorkoutRequest(BaseModel):
    Age: int
    Gender: str
    Weight: float
    Goal: str          # Weight Loss, Muscle Gain, Maintain
    Frequency: int     # 1-7
    Level: str         # Beginner, Intermediate, Advanced
    Environment: str   # Home, Gym
    
    Badminton: int = 0
    Football: int = 0
    Basketball: int = 0
    Volleyball: int = 0
    Swim: int = 0

class MealRequest(BaseModel):
    Daily_Calories: float   # Target kalori harian
    Target_Protein_g: float
    Target_Carbs_g: float
    Target_Fat_g: float
    Frequency: int = 3

class WeekRequest(BaseModel):
    Age: int
    Gender: str
    Height_cm: float
    Initial_Weight_kg: float
    Goal: str           # Muscle Gain / Weight Loss
    Level: str          # Beginner / Intermediate / Advanced
    
    # Data Tambahan
    Body_Fat_Category: int = 2 # 1=Low, 2=Avg, 3=High, 4=Very High
    Body_Fat_Percentage: float = 20.0
    
    # Data Latihan
    Frequency: int = 3
    Duration: int = 60
    
    # Hobi (Opsional)
    Badminton: int = 0
    Football: int = 0
    Basketball: int = 0
    Volleyball: int = 0
    Swim: int = 0
    Week : int

class ProgressRequest(BaseModel):
    Age: int
    Gender: str
    Height_cm: float
    Initial_Weight_kg: float
    Goal: str           # Muscle Gain / Weight Loss
    Level: str          # Beginner / Intermediate / Advanced
    
    # Data Tambahan
    Body_Fat_Category: int = 2 # 1=Low, 2=Avg, 3=High, 4=Very High
    Body_Fat_Percentage: float = 20.0
    
    # Data Latihan
    Frequency: int = 3
    Duration: int = 60
    
    # Hobi (Opsional)
    Badminton: int = 0
    Football: int = 0
    Basketball: int = 0
    Volleyball: int = 0
    Swim: int = 0

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Workout AI API is Running!"}

@app.post("/workout")
def predict_workout_endpoint(user_data: WorkoutRequest):
    """
    Endpoint ini menerima JSON dari user, lalu memanggil fungsi prediksi.
    """
    # 1. Ubah Pydantic Model ke Python Dictionary biasa
    input_dict = user_data.dict() 
    
    # 2. Panggil Logic Prediksi kita
    # (Fungsi yang tadi kita buat)
    result = get_workout_plan_json(input_dict)
    
    # 3. Cek Error
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error"))
        
    return result

@app.post("/meal")
def predict_meal_endpoint(meal_data: MealRequest):
    """
    Menerima target makro harian -> Mengembalikan rekomendasi menu makan seharian
    """
    # 1. Convert ke Dict
    input_dict = meal_data.dict()
    
    # 2. Panggil Service 'Otak' Meal
    result = get_meal_plan_json(input_dict)
    
    # 3. Handle Error
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error"))
        
    return result

# @app.post("/userprogress-byweek")
# def predict_progress_endpoint(data: ProgressRequest):
#     """
#     Memprediksi kondisi tubuh (BB, BMI, Body Fat) dan Nutrisi 
#     untuk minggu tertentu (Week X).
#     """
#     input_dict = data.dict()
#     result = get_progress_prediction(input_dict)
    
#     if result.get("status") == "failed":
#         raise HTTPException(status_code=400, detail=result.get("error"))
        
#     return result

@app.post("/userprogress")
def predict_progress_endpoint(data: ProgressRequest):
    """
    Mengembalikan Array JSON berisi progress dari Minggu 0 s.d 12
    """
    input_dict = data.dict()
    
    # Panggil fungsi looping baru kita
    result = get_progress_roadmap(input_dict)
    
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error"))
        
    return result