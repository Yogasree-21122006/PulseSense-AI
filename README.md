# 🫀 PulseSense-AI

### Smart ECG Arrhythmia Detection using Deep Learning (LSTM)

---

## 📌 Project Overview

**PulseSense-AI** is an AI-powered healthcare system that analyzes ECG (Electrocardiogram) signals to detect cardiac arrhythmias in real-time.
It uses a deep learning LSTM model to classify heart conditions and helps in early diagnosis.

---

## 🚀 Features

* 🧠 Deep Learning model (LSTM) for ECG classification
* 📊 Detects multiple arrhythmia types
* 🌐 Flask API for backend processing
* 💻 Streamlit UI for interactive frontend
* 📈 Real-time ECG signal prediction
* ⚠️ Confidence-based prediction output

---

## ❤️ Classes Detected

* **Normal** — Regular sinus rhythm
* **AFib** — Atrial Fibrillation
* **VTach** — Ventricular Tachycardia
* **PVC** — Premature Ventricular Contraction

---

## 🏗️ Project Structure

```
ecg-app/
├── app.py
├── frontend.py
├── train_model.py
├── generate_sample_data.py
├── requirements.txt
├── model/
│   ├── ecg_lstm_model.h5
│   ├── label_encoder.pkl
│   └── model_meta.pkl
└── data/
    ├── sample_normal.csv
    ├── sample_afib.csv
    ├── sample_vtach.csv
    ├── sample_pvc.csv
    └── sample_batch.csv
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Yogasree-21122006/PulseSense-AI.git
cd PulseSense-AI
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1️⃣ Train Model (Run once)

```bash
python train_model.py
```

### 2️⃣ Start Backend (Flask)

```bash
python app.py
```

### 3️⃣ Start Frontend (Streamlit)

```bash
streamlit run frontend.py --server.port 5000
```

---

## 🔌 API Endpoints

* `GET /health` → Check API status
* `POST /predict` → Predict ECG signal

### Example Response:

```json
{
  "prediction": "Normal",
  "confidence": 0.95,
  "all_probabilities": {
    "Normal": 0.95,
    "AFib": 0.02,
    "VTach": 0.01,
    "PVC": 0.02
  }
}
```

---

## 🧠 Model Architecture

* LSTM(64) → Dropout(0.3)
* LSTM(32) → Dropout(0.3)
* Dense(32) → Dense(4, Softmax)

**Loss Function:** Categorical Crossentropy
**Optimizer:** Adam

---

## 🌍 Real-World Use Case

This system can be used in:

* 🏥 Hospitals for patient monitoring
* ⌚ Wearable ECG devices
* 🚑 Emergency alert systems
* 🧑‍⚕️ Remote healthcare applications

---

## 🔮 Future Scope

* 📱 Mobile app integration
* ☁️ Cloud deployment
* 📡 Real-time IoT ECG devices
* 🚨 Emergency alert system (SMS/Email)

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Flask
* Streamlit
* NumPy, Pandas, Scikit-learn

---

## 👩‍💻 Authors

* **YOGA SREE S (24ADR185)**
---

## ⭐ Acknowledgement

This project was developed as part of an academic mini project focused on applying AI in healthcare.

---

## 📬 Contact

For queries or collaboration:
📧 Feel free to connect via GitHub

---

⭐ If you like this project, give it a star!
