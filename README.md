# 🫀 PulseSense-AI

<p align="center">
  <a href="https://ecg-arrhythmia-classifier-agkz7exbruceewwzceuxjd.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Click Here-brightgreen?style=for-the-badge">
  </a>
</p>

### Smart ECG Arrhythmia Detection using Machine Learning

---

## 📌 Project Overview

**PulseSense-AI** is an AI-powered healthcare system that analyzes ECG (Electrocardiogram) signals to detect cardiac arrhythmias in real-time.

It uses a Machine Learning model with advanced signal feature extraction to classify heart conditions and assist in early diagnosis.

🚀 The system is deployed as an interactive web application using Streamlit, enabling users to upload ECG data and get instant predictions.

---

## 🌐 Live Demo

👉 https://ecg-arrhythmia-classifier-agkz7exbruceewwzceuxjd.streamlit.app/

⚡ Upload ECG data (CSV format) and get instant arrhythmia predictions powered by AI.

> ⚠️ Note: This is a deployed Streamlit application for demonstration purposes. Model performance may vary based on input data quality.

---

## 📥 Sample Test Files

App test பண்ண இந்த files download பண்ணுங்க 👇

| File | Condition | Download |
|------|-----------|----------|
| sample_normal.csv | 💚 Normal Heart | [Download](data/sample_normal.csv) |
| sample_afib.csv | 🟡 Atrial Fibrillation | [Download](data/sample_afib.csv) |
| sample_vtach.csv | 🔴 Ventricular Tachycardia | [Download](data/sample_vtach.csv) |
| sample_pvc.csv | 🟠 Premature Ventricular Contraction | [Download](data/sample_pvc.csv) |

> 📌 Download any file → Upload in the Live Demo → Click Predict!

---

## 🚀 Features

* 🧠 ML model with advanced ECG signal feature extraction
* 📊 Detects multiple arrhythmia types
* 🌐 Fully integrated Streamlit-based web application
* 📈 Real-time ECG signal visualization
* ⚠️ Confidence-based prediction output
* 📁 CSV upload support for easy testing

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
├── requirements.txt
├── model/
│   ├── ecg_model.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── model_meta.pkl
└── data/
    ├── sample_normal.csv
    ├── sample_afib.csv
    ├── sample_vtach.csv
    └── sample_pvc.csv
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Yogasree-21122006/ecg-arrhythmia-classifier.git
cd ecg-arrhythmia-classifier
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run frontend.py
```

---

## 🧠 Model Details

**Feature Extraction from ECG Signal:**
* Statistical features — Mean, Std, Min, Max, RMS, Percentiles
* Temporal features — First & Second order differences
* Frequency features — FFT-based spectral analysis
* Segment features — 5-segment std & range analysis
* Morphological features — Zero crossings, Peak count, Peak interval std

**Pipeline:**
* Feature Extraction → Standard Scaling → ML Classifier → Softmax Probabilities

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
* 🧠 Deep Learning (LSTM) model upgrade

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Streamlit
* NumPy, Pandas, Matplotlib, SciPy

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

⭐ If you like this project, don't forget to give it a star!
