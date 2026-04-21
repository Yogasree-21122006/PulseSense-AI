
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
model = None
scaler = None
label_encoder = None
model_meta = None


def extract_features(signal):
    signal = np.array(signal, dtype=float)
    feats = []
    feats.append(np.mean(signal))
    feats.append(np.std(signal))
    feats.append(np.max(signal))
    feats.append(np.min(signal))
    feats.append(np.max(signal) - np.min(signal))
    feats.append(np.sqrt(np.mean(signal ** 2)))
    feats.extend(np.percentile(signal, [10, 25, 50, 75, 90]).tolist())
    diff1 = np.diff(signal)
    feats.append(np.mean(np.abs(diff1)))
    feats.append(np.std(diff1))
    diff2 = np.diff(diff1)
    feats.append(np.mean(np.abs(diff2)))
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    total_power = np.sum(fft_vals ** 2) + 1e-12
    feats.append(np.sum(freqs * fft_vals ** 2) / total_power)
    feats.extend(fft_vals[:10].tolist())
    n_segs = 5
    seg_len = len(signal) // n_segs
    for i in range(n_segs):
        seg = signal[i * seg_len:(i + 1) * seg_len]
        feats.append(np.std(seg))
        feats.append(np.max(seg) - np.min(seg))
    crossings = np.where(np.diff(np.sign(signal)))[0]
    feats.append(len(crossings))
    peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0]
    feats.append(len(peaks))
    if len(peaks) > 1:
        feats.append(np.std(np.diff(peaks)))
    else:
        feats.append(0.0)
    return np.array(feats, dtype=float)


def load_model():
    global model, scaler, label_encoder, model_meta
    try:
        model_path = os.path.join(MODEL_DIR, 'ecg_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
        meta_path = os.path.join(MODEL_DIR, 'model_meta.pkl')

        if not os.path.exists(model_path):
            return False, "Model file not found. Please train the model first by running: python3.11 train_model.py"

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(meta_path, 'rb') as f:
            model_meta = pickle.load(f)

        return True, "Model loaded successfully"
    except Exception as e:
        return False, str(e)


def preprocess_signal(signal_values, window_size=187):
    signal = np.array(signal_values, dtype=float)
    if len(signal) > window_size:
        signal = signal[:window_size]
    elif len(signal) < window_size:
        signal = np.pad(signal, (0, window_size - len(signal)), mode='edge')
    mean = signal.mean()
    std = signal.std() + 1e-8
    signal = (signal - mean) / std
    return signal


@app.route('/health', methods=['GET'])
def health():
    model_loaded = model is not None
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_type': model_meta.get('model_type') if model_meta else None,
        'test_accuracy': model_meta.get('test_accuracy') if model_meta else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        success, msg = load_model()
        if not success:
            return jsonify({'error': f'Model not loaded: {msg}'}), 503

    try:
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file, header=None)
            signal_values = df.values.flatten().tolist()
        elif request.is_json:
            data = request.get_json()
            signal_values = data.get('signal', [])
            if not signal_values:
                return jsonify({'error': 'No signal data provided'}), 400
        else:
            return jsonify({'error': 'No data provided. Send CSV file or JSON with signal array.'}), 400

        window_size = model_meta.get('window_size', 187) if model_meta else 187
        signal = preprocess_signal(signal_values, window_size)
        features = extract_features(signal).reshape(1, -1)
        features_scaled = scaler.transform(features)

        proba = model.predict_proba(features_scaled)[0]
        pred_idx = np.argmax(proba)
        confidence = float(proba[pred_idx])
        class_name = label_encoder.classes_[pred_idx]

        all_probs = {
            str(label_encoder.classes_[i]): float(proba[i])
            for i in range(len(label_encoder.classes_))
        }

        return jsonify({
            'prediction': class_name,
            'confidence': round(confidence, 4),
            'all_probabilities': all_probs,
            'signal_length': len(signal_values)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    if label_encoder is None:
        load_model()
    classes = list(label_encoder.classes_) if label_encoder else ['Normal', 'AFib', 'VTach', 'PVC']
    return jsonify({'classes': classes})


if __name__ == '__main__':
    print("Loading model...")
    success, msg = load_model()
    print(f"Model load: {msg}")
    port = int(os.environ.get('FLASK_PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
