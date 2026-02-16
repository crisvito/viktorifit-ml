# Viktorifit Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-grey?style=flat&logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)

**Viktorifit ML** adalah repositori yang menangani seluruh proses Machine Learning untuk ekosistem aplikasi Viktorifit.

Sistem ini dirancang untuk memberikan rekomendasi kesehatan yang dipersonalisasi (makanan, rencana latihan, dan prediksi progres tubuh) dengan memanfaatkan *Machine Learning*. Kami fokus pada **latensi rendah** (*low-latency*) dengan mengoptimalkan model regresi ke format **ONNX** untuk kebutuhan produksi.

## Daftar Isi
- [Struktur Proyek](#-struktur-proyek)
- [Fitur Utama](#-fitur-utama)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Deployment](#-deployment)
- [Kontribusi](#-kontribusi)

## Struktur Proyek

Berikut adalah struktur direktori dari *source code* ini:

```bash
viktorifit-ml/
├── data/                      # Dataset mentah & hasil pemrosesan
│   ├── dataset_meal.csv
│   ├── dataset_users.csv
│   ├── dataset_workout_final.csv
│   └── ...
├── models/                    # Model Registry
│   ├── onnx/                  # Model teroptimasi untuk production
│   │   ├── Body_Fat_Percentage_y.onnx
│   │   ├── Daily_Calories.onnx
│   │   ├── Target_Protein_g.onnx
│   │   ├── Daily_Water_ml.onnx
│   │   └── ...
│   ├── model_meal.pickle      # Model rekomendasi (Scikit-learn object)
│   ├── model_workout.pickle
│   └── model_progress.pickle
├── notebooks/                 # Jupyter Notebooks untuk eksperimen & training
│   ├── modeling_dataset_meal.ipynb
│   ├── modeling_dataset_workout.ipynb
│   └── test_predict/          # Script pengujian inferensi
│       ├── predict_meal.ipynb
│       └── predict_workout.ipynb
├── requirements.txt           # Daftar dependensi Python
└── README.md
```

## Fitur Utama

### 1. Nutritional & Body Forecasting (ONNX)

Model regresi yang dioptimalkan untuk **latensi rendah (low-latency)** menggunakan format **ONNX**. Fitur ini memprediksi:

**Kebutuhan Energi & Cairan:**

* Kalori Harian (`Daily_Calories.onnx`)
* Target Air Minum (`Daily_Water_ml.onnx`)

**Target Makro & Mikro Nutrisi:**

* Protein (`Target_Protein_g.onnx`)
* Karbohidrat (`Target_Carbs_g.onnx`)
* Lemak (`Target_Fat_g.onnx`)
* Serat (`Target_Fiber_g.onnx`)
* Batas Gula Harian (`Limit_Sugar_g.onnx`)

**Metrik Tubuh:**

* Estimasi Persentase Lemak Tubuh (`Body_Fat_Percentage_y.onnx`)
* Prediksi Berat Badan (`Weight_kg.onnx`)

---

### 2. Workout & Progress Engine (Pickle)

Sistem cerdas berbasis machine learning untuk aktivitas fisik:

**Workout Recommender (`model_workout.pickle`)**
Memberikan rekomendasi rencana latihan (jenis olahraga, durasi, intensitas) yang disesuaikan dengan tujuan pengguna (misalnya: *Weight Loss*, *Muscle Gain*, atau *Endurance*).

**User Progress Tracker (`model_progress.pickle`)**
Menganalisis riwayat latihan dan perkembangan fisik pengguna untuk memprediksi pencapaian target atau menyesuaikan tingkat kesulitan latihan selanjutnya.

---

### 3. Meal Recommendation System (Pickle)

**Meal Planner (`model_meal.pickle`)**
Menyarankan paket menu makanan yang sesuai dengan target kalori harian dan preferensi diet pengguna.

---

## Dataset

Dataset yang digunakan dalam proyek ini mencakup:

* **Nutrisi:** Database makanan dengan rincian makro (Protein, Karbohidrat, Lemak) dan mikro (Serat, Gula).
* **User Logs:** Data simulasi profil pengguna (Tinggi, Berat, Aktivitas) untuk melatih model regresi tubuh.
* **Workout & Progress:** Database jenis latihan fisik dan log perkembangan pengguna untuk analisis performa.

---

## Instalasi

Pastikan Anda telah menginstal **Python 3.8 atau versi yang lebih baru**.

### 1. Clone Repositori

```bash
git clone https://github.com/crisvito/viktorifit-ml.git
cd viktorifit-ml
```

### 2. Buat Virtual Environment (Sangat Disarankan)

Agar dependencies tidak bentrok dengan sistem utama.

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Penggunaan

### 1. Menjalankan Training (Pelatihan Ulang)

Jika Anda ingin melatih ulang model dengan data baru, jalankan notebook yang relevan di folder `notebooks/`.

Contoh untuk melatih model workout:

```bash
jupyter notebook notebooks/modeling_dataset_workout.ipynb
```

### 2. Contoh Inferensi (Prediksi ONNX)

Berikut adalah contoh skrip Python untuk memuat model ONNX dan melakukan prediksi tanpa menjalankan API penuh:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/onnx/Daily_Calories.onnx")

input_data = np.array([[25, 170, 65]], dtype=np.float32)  # contoh input
result = session.run(None, {"input": input_data})

print("Prediksi:", result)
```

---

## Deployment

Model-model dalam repositori ini dirancang untuk di-*serve* menggunakan kerangka kerja API seperti **FastAPI**.

* **Model ONNX:** Dimuat menggunakan `onnxruntime` untuk performa CPU yang cepat dan ringan.
* **Model Pickle:** Dimuat menggunakan `pickle` atau `joblib` untuk logika rekomendasi yang kompleks.

Pastikan server produksi memiliki library yang tercantum di `requirements.txt` terinstal.
