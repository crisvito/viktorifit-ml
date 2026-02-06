import pickle
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'models', 'model_progress.pickle')

print("⏳ Loading Progress Model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        data_model = pickle.load(f)
        
    progress_models = data_model['models_dict'] # Dictionary berisi banyak model (Weight, BMI, Calories, dll)
    progress_encoders = data_model['encoders']
    progress_features = data_model['features']
    
    print("✅ Model Progress Berhasil Diload!")
    
except Exception as e:
    print(f"❌ Error Load Model Progress: {e}")
    progress_models = None

# ==========================================
# 2. FUNGSI LOGIKA (SERVICE)
# ==========================================
def get_progress_prediction(user_input: dict):
    """
    Menerima data user dan minggu ke-X.
    Mengembalikan prediksi kondisi fisik dan kebutuhan nutrisi pada minggu tersebut.
    """
    
    if progress_models is None:
        return {"status": "failed", "error": "Model Progress belum siap."}

    try:
        # --- A. PREPROCESSING (HITUNG DATA TURUNAN) ---
        # Kita hitung BMI dan Kategorinya secara otomatis di sini biar User gak perlu input manual
        tinggi_cm = user_input.get('Height_cm', 170)
        berat_awal = user_input.get('Initial_Weight_kg', 60)
        
        tinggi_m = tinggi_cm / 100
        bmi_awal = round(berat_awal / (tinggi_m ** 2), 2)
        
        # Logika Kategori BMI (Standar WHO)
        if bmi_awal < 18.5: cat_bmi = 'Underweight'
        elif bmi_awal < 25: cat_bmi = 'Normal'
        elif bmi_awal < 30: cat_bmi = 'Overweight'
        else: cat_bmi = 'Obese'

        # --- B. ENCODING INPUT ---
        # Mengubah teks user menjadi angka yang dimengerti model
        gender_code = progress_encoders['Gender'].transform([user_input.get('Gender', 'Male')])[0]
        goal_code = progress_encoders['Goal'].transform([user_input.get('Goal', 'Muscle Gain')])[0]
        level_code = progress_encoders['level'].transform([user_input.get('Level', 'Beginner')])[0]
        bmi_cat_code = progress_encoders['BMI_Category_x'].transform([cat_bmi])[0]
        
        # --- C. SUSUN ARRAY INPUT (SESUAI URUTAN FITUR MODEL) ---
        input_row = [
            user_input.get('Age', 25),
            gender_code,
            tinggi_cm,
            berat_awal,
            bmi_awal,
            bmi_cat_code,
            user_input.get('Body_Fat_Category', 2), # Default 2 (Average)
            user_input.get('Body_Fat_Percentage', 15.0),
            goal_code,
            user_input.get('Frequency', 3),
            user_input.get('Duration', 60),
            level_code,
            # Sports
            int(user_input.get('Badminton', 0)),
            int(user_input.get('Football', 0)),
            int(user_input.get('Basketball', 0)),
            int(user_input.get('Volleyball', 0)),
            int(user_input.get('Swim', 0)),
            # PENTING: Minggu ke berapa yang mau diprediksi
            user_input.get('Week', 1) 
        ]
        
        # Convert ke DataFrame
        input_df = pd.DataFrame([input_row], columns=progress_features)

        # --- D. LOOP PREDIKSI (MULTI-OUTPUT) ---
        # Model progress kamu terdiri dari banyak sub-model (untuk berat, lemak, kalori, dll)
        hasil_prediksi = {}
        
        for target_name, info in progress_models.items():
            model = info['model']
            tipe_model = info['type'] 
            
            # Prediksi (Hasilnya masih NumPy Array)
            pred_array = model.predict(input_df)
            
            # Ambil nilai pertama (Hasilnya masih numpy.float32 / numpy.int)
            pred_value = pred_array[0]
            
            if tipe_model == 'categorical':
                # Decode: Ubah angka kembali jadi teks
                original_col_name = target_name.replace('_Encoded', '')
                
                # Pastikan input ke inverse_transform adalah integer
                decoded_text = progress_encoders[original_col_name].inverse_transform([int(pred_value)])[0]
                hasil_prediksi[original_col_name] = decoded_text
            else:
                # --- PERBAIKAN UTAMA DI SINI ---
                # Konversi Paksa dari NumPy ke Python Native Type
                val_native = float(pred_value) 
                
                if 'mg' in target_name or 'Calories' in target_name or 'ml' in target_name:
                    # Konversi ke Integer Python
                    hasil_prediksi[target_name] = int(val_native)
                else:
                    # Konversi ke Float Python (sudah float dari baris val_native)
                    hasil_prediksi[target_name] = round(val_native, 2)

        # --- E. FORMAT JSON RESPONSE ---
        return {
            "status": "success",
            "week": user_input.get('Week', 1),
            "physical_projection": {
                "weight_kg": hasil_prediksi.get('Weight_kg'),
                "bmi": hasil_prediksi.get('BMI'),
                "bmi_category": hasil_prediksi.get('BMI_Category_y'),
                "body_fat_percentage": hasil_prediksi.get('Body_Fat_Percentage_y')
            },
            "daily_nutrition": {
                "calories": hasil_prediksi.get('Daily_Calories'),
                "water_ml": hasil_prediksi.get('Daily_Water_ml'),
                "sugar_limit_g": hasil_prediksi.get('Limit_Sugar_g'),
                "cholesterol_limit_mg": hasil_prediksi.get('Limit_Cholesterol_mg')
            },
            "macro_nutrients": {
                "protein_g": hasil_prediksi.get('Target_Protein_g'),
                "carbs_g": hasil_prediksi.get('Target_Carbs_g'),
                "fat_g": hasil_prediksi.get('Target_Fat_g'),
                "fiber_g": hasil_prediksi.get('Target_Fiber_g'),
                "calcium_mg": hasil_prediksi.get('Target_Calcium_mg')
            }
        }

    except Exception as e:
        return {"status": "failed", "error": f"Error during progress prediction: {str(e)}"}
    
def get_progress_roadmap(user_input: dict):
    """
    Menghasilkan list prediksi dari Minggu ke-0 sampai Minggu ke-12.
    """
    
    if progress_models is None:
        return {"status": "failed", "error": "Model Progress belum siap."}

    roadmap = []
    
    # Loop dari Minggu 0 sampai 12
    for week in range(0, 13): # range(0, 13) artinya 0 s/d 12
        
        # 1. Update 'Week' di input data
        # Kita copy input user agar tidak merusak data asli, lalu set minggunya
        current_input = user_input.copy()
        current_input['Week'] = week
        
        # 2. Panggil fungsi prediksi SATUAN yang sudah ada
        # (Kita gunakan fungsi get_progress_prediction yang sudah kita buat sebelumnya)
        single_result = get_progress_prediction(current_input)
        
        # 3. Cek jika gagal
        if single_result.get("status") == "failed":
            return single_result # Return error langsung
            
        # 4. Ambil datanya saja (tanpa status 'success' yang berulang)
        # Kita rapikan sedikit formatnya untuk list
        daily_nutrition = single_result['daily_nutrition']
        macro = single_result['macro_nutrients']
        physical = single_result['physical_projection']
        
        summary = {
            "week": week,
            "physical": physical,
            "nutrition": daily_nutrition,
            "macro": macro
        }
        
        roadmap.append(summary)
        
    # Kembalikan List Lengkap
    return {
        "status": "success",
        "total_weeks": 12,
        "roadmap": roadmap
    }