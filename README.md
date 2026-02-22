# ⚡ Viktorifit Machine Learning

[![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://viktorifit.vercel.app)
[![Railway](https://img.shields.io/badge/Railway-131415?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)

**Viktorifit ML** adalah pusat kecerdasan buatan yang menggerakkan ekosistem kesehatan Viktorifit. Sistem ini menyediakan rekomendasi nutrisi, rencana latihan, dan prediksi perkembangan tubuh yang dipersonalisasi dengan fokus pada **latensi rendah (low-latency)** menggunakan format **ONNX**.

---

## ✨ Features

### 🎯 Core Functionality
* **Nutritional Forecasting (ONNX):** Prediksi kebutuhan kalori, air harian, serta makro/mikro nutrisi (Protein, Karbo, Lemak, Serat, Gula) secara real-time.
* **Body Metrics Prediction:** Estimasi persentase lemak tubuh dan proyeksi berat badan masa depan.
* **Workout Recommender (Pickle):** Sistem cerdas yang menentukan jenis olahraga, durasi, dan intensitas berdasarkan profil *Weight Loss, Muscle Gain,* atau *Endurance*.
* **Meal Planner:** Personalisasi paket menu makanan yang selaras dengan target nutrisi harian pengguna.

### 🚀 Optimization & Performance
* **ONNX Integration:** Model regresi dioptimalkan untuk eksekusi CPU yang ringan dan cepat pada lingkungan produksi.
* **Progress Tracking:** Analisis riwayat latihan untuk memprediksi pencapaian target pengguna secara dinamis.
* **Clean Data Pipeline:** Manajemen dataset yang terstruktur untuk proses pelatihan ulang (retraining) yang efisien.

---

## 📂 Project Structure

```text
viktorifit-ml/
├── 📁 data/                      # Dataset mentah & hasil pemrosesan
│   ├── dataset_meal.csv
│   ├── dataset_users.csv
│   ├── dataset_workout_final.csv
│   └── ...
├── 📁 models/                    # Model Registry
│   ├── 📁 onnx/                  # Model teroptimasi untuk production
│   │   ├── Body_Fat_Percentage_y.onnx
│   │   ├── Daily_Calories.onnx
│   │   ├── Target_Protein_g.onnx
│   │   └── ...
│   ├── model_meal.pickle      # Model rekomendasi (Scikit-learn object)
│   ├── model_workout.pickle
│   └── model_progress.pickle
├── 📁 notebooks/                 # Jupyter Notebooks untuk eksperimen & training
│   ├── modeling_dataset_meal.ipynb
│   ├── modeling_dataset_workout.ipynb
│   └── 📁 test_predict/          # Script pengujian inferensi
│       ├── predict_meal.ipynb
│       └── predict_workout.ipynb
├── 📄 requirements.txt           # Daftar dependensi Python
└── 📄 README.md                  # Dokumentasi proyek

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or later
* Virtual environment (Recommended)
* Git

### Installation

1. **Clone the repository**
```bash
git clone [https://github.com/crisvito/viktorifit-ml.git](https://github.com/crisvito/viktorifit-ml.git)
cd viktorifit-ml

```


2. **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# atau venv\Scripts\activate untuk Windows

```

3. **Install Dependencies**
```bash
pip install -r requirements.txt

```

---

## 🎮 How to Use

1. **Model Training:** Jalankan Jupyter Notebook di folder `notebooks/` untuk melatih ulang model dengan data baru.
2. **Inference (ONNX):** Gunakan skrip berikut untuk melakukan prediksi cepat:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/onnx/Daily_Calories.onnx")
input_data = np.array([[25, 170, 65]], dtype=np.float32)
result = session.run(None, {"input": input_data})
print("Prediksi Kalori:", result)

```
---

## 🛠️ Technical Details

### Tech Stack

* **Modeling:** Scikit-learn, XGBoost, Pickle model, ONNX Runtime
* **Data Processing:** Pandas, NumPy
* **Inference Engine:** FastAPI (Serving API)
* **Experimentation:** Jupyter Notebook

### 🚀 Deployment

* **Frontend (Vercel):** Interface aplikasi di-host di [viktorifit.vercel.app](https://viktorifit.vercel.app).
* **Backend API (Railway):** Server API untuk model Machine Learning di-deploy melalui Railway untuk menjamin skalabilitas dan performa tinggi.

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

**Made with ❤️ by Viktorifit Team**
*Transforming health data into personalized fitness journeys!* 🍎💪✨

```

---
