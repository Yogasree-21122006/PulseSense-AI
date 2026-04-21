
import numpy as np
import pandas as pd
import os

np.random.seed(123)

CLASSES = ['Normal', 'AFib', 'VTach', 'PVC']
WINDOW_SIZE = 187
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_ecg_signal(class_name, n_samples=WINDOW_SIZE):
    t = np.linspace(0, 2 * np.pi, n_samples)
    if class_name == 'Normal':
        signal = (
            0.1 * np.sin(2 * t) +
            1.0 * np.exp(-((t - np.pi/2) % (2*np.pi/1.2) - 0.5)**2 / 0.02) +
            0.3 * np.exp(-((t - np.pi/2 + 0.15) % (2*np.pi/1.2) - 0.5)**2 / 0.05) +
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
            0.8 * np.exp(-((t - np.pi/2) % (2*np.pi/1.2) - 0.5)**2 / 0.02) +
            np.random.normal(0, 0.02, n_samples)
        )
        for pos in [45, 90, 135]:
            if pos < n_samples:
                signal[pos] = -1.5 + np.random.normal(0, 0.1)
                if pos + 5 < n_samples:
                    signal[pos+5] = 1.2 + np.random.normal(0, 0.1)
    return signal


for cls in CLASSES:
    signal = generate_ecg_signal(cls)
    df = pd.DataFrame(signal, columns=['amplitude'])
    fname = os.path.join(OUTPUT_DIR, f'sample_{cls.lower()}.csv')
    df.to_csv(fname, index=False, header=False)
    print(f"Saved: {fname}")

batch_rows = []
for cls in CLASSES:
    for _ in range(5):
        sig = generate_ecg_signal(cls)
        batch_rows.append(sig.tolist())

batch_df = pd.DataFrame(batch_rows)
batch_path = os.path.join(OUTPUT_DIR, 'sample_batch.csv')
batch_df.to_csv(batch_path, index=False, header=False)
print(f"Saved batch: {batch_path}")
print("Sample data generation complete.")
