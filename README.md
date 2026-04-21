# 🫀 PulseSense-AI

<p align="center">
  <a href="https://ecg-arrhythmia-classifier-agkz7exbruceewwzceuxjd.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Click Here-brightgreen?style=for-the-badge">
  </a>
</p>

### Smart ECG Arrhythmia Detection using Deep Learning (LSTM)

---

## 📌 Project Overview

**PulseSense-AI** is an AI-powered healthcare system that analyzes ECG (Electrocardiogram) signals to detect cardiac arrhythmias in real-time.
It uses a deep learning LSTM model to classify heart conditions and assist in early diagnosis.

🚀 The system is deployed as an interactive web application using Streamlit, enabling users to upload ECG data and get instant predictions.

---

## 🌐 Live Demo

👉 https://ecg-arrhythmia-classifier-agkz7exbruceewwzceuxjd.streamlit.app/

⚡ Upload ECG data (CSV format) and get instant arrhythmia predictions powered by AI.

> ⚠️ Note: This is a deployed Streamlit application for demonstration purposes. Model performance may vary based on input data quality.

---

## 🚀 Features

* 🧠 Deep Learning model (LSTM) for ECG classification
* 📊 Detects multiple arrhythmia types
* 🌐 Fully integrated Streamlit-based application (Frontend + Backend)
* 📈 Real-time ECG signal prediction
* ⚠️ Confidence-based prediction output
* 📁 CSV upload support for batch analysis

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

### 1️⃣ Train Model (Optional - Run once)

```bash
python train_model.py
```

### 2️⃣ Run the Application

```bash
streamlit run frontend.py
```

---

## 🧠 Model Architecture

* LSTM(64) → Dropout(0.3)
* LSTM(32) → Dropout(0.3)
* Dense(32) → Dense(4, Softmax)

**Loss Function:** Categorical Crossentropy
**Optimizer:** Adam

---

## 🌍 Real-World Use Cases

* 🏥 Hospital patient monitoring systems
* ⌚ Wearable ECG health devices
* 🚑 Emergency detection and alert systems
* 🧑‍⚕️ Remote healthcare & telemedicine

---

## 🔮 Future Scope

* 📱 Mobile app integration
* ☁️ Cloud-based scaling
* 📡 Real-time IoT ECG device integration
* 🚨 Automated emergency alerts (SMS/Email)

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* NumPy, Pandas, Scikit-learn

---

## 👩‍💻 Author

* **YOGA SREE S (24ADR185)**

---

## ⭐ Acknowledgement

This project was developed as part of an academic mini project focused on applying AI in healthcare.

---

## 📬 Contact

For queries or collaboration:
📧 Feel free to connect via GitHub

---

⭐ If you like this project, don’t forget to give it a star!
