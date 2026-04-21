# ECG Arrhythmia Classification System

AI-powered ECG signal analysis using LSTM deep learning for arrhythmia detection.

## Classes
- **Normal** — Regular sinus rhythm
- **AFib** — Atrial Fibrillation
- **VTach** — Ventricular Tachycardia
- **PVC** — Premature Ventricular Contraction

## Project Structure
```
ecg-app/
├── app.py                  # Flask backend API
├── frontend.py             # Streamlit UI
├── train_model.py          # Model training script
├── generate_sample_data.py # Generate test CSV files
├── requirements.txt        # Python dependencies
├── model/                  # Trained model files
│   ├── ecg_lstm_model.h5
│   ├── label_encoder.pkl
│   └── model_meta.pkl
└── data/                   # Sample ECG CSV files
    ├── sample_normal.csv
    ├── sample_afib.csv
    ├── sample_vtach.csv
    ├── sample_pvc.csv
    └── sample_batch.csv
```

## Running the System

1. **Train the model** (run once):
   ```bash
   python train_model.py
   ```

2. **Start Flask backend**:
   ```bash
   python app.py
   ```

3. **Start Streamlit frontend**:
   ```bash
   streamlit run frontend.py --server.port 5000
   ```

## API Endpoints

- `GET /health` — Check server and model status
- `POST /predict` — Classify ECG signal
  - Accepts: CSV file upload or JSON `{"signal": [...]}`
  - Returns: `{"prediction": "Normal", "confidence": 0.95, "all_probabilities": {...}}`

## Model Architecture

LSTM model trained on 187-sample ECG windows:
- LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(32) → Dense(4, softmax)
- Loss: Categorical Cross-Entropy
- Optimizer: Adam
