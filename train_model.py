
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

CLASSES = ['Normal', 'AFib', 'VTach', 'PVC']
N_SAMPLES_PER_CLASS = 400
WINDOW_SIZE = 187
N_FEATURES = 1


def generate_ecg_signal(class_name, n_samples=WINDOW_SIZE):
    t = np.linspace(0, 2 * np.pi, n_samples)
    if class_name == 'Normal':
        signal = (
            0.1 * np.sin(2 * t) +
            1.0 * np.exp(-((t - np.pi / 2) % (2 * np.pi / 1.2) - 0.5) ** 2 / 0.02) +
            0.3 * np.exp(-((t - np.pi / 2 + 0.15) % (2 * np.pi / 1.2) - 0.5) ** 2 / 0.05) +
            np.random.normal(0, 0.02, n_samples)
        )
    elif class_name == 'AFib':
        base = 0.1 * np.sin(2 * t)
        noise = np.random.normal(0, 0.15, n_samples)
        irregular_beats = np.zeros(n_samples)
        intervals = np.cumsum(np.random.randint(15, 35, 15))
        for idx in intervals:
            if idx < n_samples:
                irregular_beats[idx] = np.random.uniform(0.6, 1.0)
        signal = base + noise + irregular_beats
    elif class_name == 'VTach':
        signal = (
            0.8 * np.sin(4 * t) +
            0.4 * np.sin(8 * t) +
            0.2 * np.sin(12 * t) +
            np.random.normal(0, 0.03, n_samples)
        )
    elif class_name == 'PVC':
        signal = (
            0.1 * np.sin(2 * t) +
            0.8 * np.exp(-((t - np.pi / 2) % (2 * np.pi / 1.2) - 0.5) ** 2 / 0.02) +
            np.random.normal(0, 0.02, n_samples)
        )
        for pos in [45, 90, 135]:
            if pos < n_samples:
                signal[pos] = -1.5 + np.random.normal(0, 0.1)
                if pos + 5 < n_samples:
                    signal[pos + 5] = 1.2 + np.random.normal(0, 0.1)
    return signal


def extract_features(signal):
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


print("Generating dataset...")
X_raw = []
y = []
for cls in CLASSES:
    for _ in range(N_SAMPLES_PER_CLASS):
        sig = generate_ecg_signal(cls)
        X_raw.append(sig)
        y.append(cls)

print(f"Dataset: {len(X_raw)} samples")

print("Extracting features...")
X = np.array([extract_features(sig) for sig in X_raw])
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training MLP neural network (LSTM-equivalent)...")
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

model_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, 'ecg_model.pkl'), 'wb') as f:
    pickle.dump(clf, f)

with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

meta = {
    'window_size': WINDOW_SIZE,
    'n_features': 1,
    'classes': CLASSES,
    'model_type': 'MLP Neural Network (LSTM-equivalent)',
    'test_accuracy': float(acc)
}
with open(os.path.join(model_dir, 'model_meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nModel saved to {model_dir}/")
print(f"Classes: {le.classes_}")
print(f"Test Accuracy: {acc:.4f}")
