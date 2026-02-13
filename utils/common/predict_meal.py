import pickle
import pandas as pd
import numpy as np
import os
# ==========================================
# 1. LOAD MODEL MEAL
# ==========================================
# Sesuaikan path jika berbeda
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models', 'model_meal.pickle')

print("Loading Meal Model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        meal_data = pickle.load(f)
        
    knn_meal = meal_data['knn_model']
    scaler_meal = meal_data['scaler']
    db_meal = meal_data['meal_db']
    features_meal = meal_data['features'] # Kolom yg diharapkan: ['Energy', 'Protein', 'Carbs', 'Fat']
    
    print("Model Meal Berhasil Diload!")
    
except Exception as e:
    print(f"Error Load Model Meal: {e}")
    knn_meal = None

# ==========================================
# 2. FUNGSI LOGIKA (SERVICE)
# ==========================================
def get_meal_plan_json(user_targets: dict):
    """
    Menerima target nutrisi harian user, 
    mengembalikan rencana makan lengkap dengan porsi yang disesuaikan.
    """
    
    if knn_meal is None:
        return {"status": "failed", "error": "Model Meal belum siap."}

    # --- A. AMBIL DATA INPUT ---
    freq_makan = user_targets.get('Frequency', 3)
    target_cal = user_targets.get('Daily_Calories', 2000)
    target_prot = user_targets.get('Target_Protein_g', 150)
    target_carbs = user_targets.get('Target_Carbs_g', 200)
    target_fat = user_targets.get('Target_Fat_g', 60)

    # --- B. HITUNG TARGET PER MAKAN ---
    avg_cal = target_cal / freq_makan
    avg_prot = target_prot / freq_makan
    avg_carbs = target_carbs / freq_makan
    avg_fat = target_fat / freq_makan
    
    # --- C. SIAPKAN DATA UNTUK MODEL ---
    # Mapping input ke fitur yang dikenali model
    # Pastikan nama key ini SAMA PERSIS dengan 'features_meal' di model kamu
    input_map = {
        'Energy': avg_cal,
        'Protein': avg_prot,
        'Carbs': avg_carbs,
        'Fat': avg_fat,
    }
    
    # Buat DataFrame & Scale
    try:
        input_df = pd.DataFrame([input_map])[features_meal]
        input_scaled = scaler_meal.transform(input_df)
    except KeyError as e:
        return {"status": "failed", "error": f"Fitur model tidak cocok: {str(e)}"}
    except Exception as e:
        return {"status": "failed", "error": f"Error scaling data: {str(e)}"}

    # --- D. LOOP PENCARIAN MENU ---
    menu_plan = []
    menu_names_taken = [] # Untuk cek duplikat
    
    total_planned_cal = 0
    total_planned_prot = 0
    total_planned_carbs = 0
    total_planned_fat = 0

    for i in range(1, freq_makan + 1):
        # 1. Cari 10 kandidat terdekat
        distances, indices = knn_meal.kneighbors(input_scaled, n_neighbors=10)
        candidates = db_meal.iloc[indices[0]].copy()
        
        # 2. Filter Duplikat (Jangan makan menu yang sama dalam sehari)
        candidates = candidates[~candidates['Food Items'].isin(menu_names_taken)]
        
        # Kalau habis difilter malah kosong, reset ambil lagi kandidat awal
        if candidates.empty:
            candidates = db_meal.iloc[indices[0]].copy()
            
        # 3. Pilih Top 1
        pilihan = candidates.iloc[0]
        
        # 4. SMART PORTIONING LOGIC
        # Rumus: Target Kalori per Sesi / Kalori Makanan Asli
        raw_porsi = avg_cal / pilihan['Energy']
        
        # Bulatkan ke 0.5 terdekat (0.5, 1.0, 1.5, 2.0) biar masuk akal
        porsi = round(raw_porsi * 2) / 2
        
        # Safety limit (Jangan terlalu sedikit atau terlalu rakus)
        if porsi < 0.5: porsi = 0.5
        if porsi > 3.0: porsi = 3.0 
        
        # 5. Hitung Nutrisi Real
        real_cal = pilihan['Energy'] * porsi
        real_prot = pilihan['Protein'] * porsi
        real_carbs = pilihan['Carbs'] * porsi
        real_fat = pilihan['Fat'] * porsi
        
        # 6. Simpan ke List
        menu_item = {
            "meal_order": i,
            "menu_name": pilihan['Food Items'],
            "portion": porsi,
            "calories": round(real_cal, 1),
            "protein": round(real_prot, 1),
            "carbs": round(real_carbs, 1),
            "fat": round(real_fat, 1)
        }
        
        menu_plan.append(menu_item)
        menu_names_taken.append(pilihan['Food Items'])
        
        # Akumulasi Total Harian
        total_planned_cal += real_cal
        total_planned_prot += real_prot
        total_planned_carbs += real_carbs
        total_planned_fat += real_fat

    # --- E. RETURN HASIL JSON ---
    return {
        "status": "success",
        "target_daily": {
            "calories": target_cal,
            "protein": target_prot,
            "carbs": target_carbs,
            "fat": target_fat
        },
        "planned_total": {
            "calories": round(total_planned_cal, 1),
            "protein": round(total_planned_prot, 1),
            "carbs": round(total_planned_carbs, 1),
            "fat": round(total_planned_fat, 1)
        },
        "meal_plan": menu_plan
    }