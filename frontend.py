
import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
import json

BACKEND_URL = os.environ.get('FLASK_BACKEND_URL', 'http://localhost:5001')
st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 ECG Arrhythmia Classification System")
st.markdown("**AI-powered ECG signal analysis using LSTM deep learning**")
st.markdown("---")

CLASS_DESCRIPTIONS = {
    'Normal': {
        'desc': 'Regular heartbeat with normal sinus rhythm.',
        'color': '#2ecc71',
        'severity': 'Healthy'
    },
    'AFib': {
        'desc': 'Atrial Fibrillation — irregular, rapid atrial activity.',
        'color': '#e67e22',
        'severity': 'Moderate Risk'
    },
    'VTach': {
        'desc': 'Ventricular Tachycardia — rapid ventricular rhythm, potentially life-threatening.',
        'color': '#e74c3c',
        'severity': 'High Risk'
    },
    'PVC': {
        'desc': 'Premature Ventricular Contraction — extra heartbeat from the ventricle.',
        'color': '#f39c12',
        'severity': 'Low-Moderate Risk'
    }
}


def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.json()
    except Exception as e:
        return None


def plot_ecg_signal(signal_values, title="ECG Signal", prediction=None):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    color = '#2c3e50'
    if prediction and prediction in CLASS_DESCRIPTIONS:
        color = CLASS_DESCRIPTIONS[prediction]['color']

    ax.plot(signal_values, color=color, linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Amplitude (normalized)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    return fig


def plot_probabilities(all_probs):
    classes = list(all_probs.keys())
    probs = [all_probs[c] * 100 for c in classes]
    colors = [CLASS_DESCRIPTIONS.get(c, {}).get('color', '#95a5a6') for c in classes]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(classes, probs, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold'
        )

    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Classification Probabilities", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def generate_sample_ecg(class_name='Normal', n_samples=187):
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
                signal[pos] = -1.5
                if pos + 5 < n_samples:
                    signal[pos + 5] = 1.2
    return signal


with st.sidebar:
    st.header("System Status")
    status = check_backend()
    if status:
        st.success("Backend: Connected")
        st.info(f"Model: {'Loaded' if status.get('model_loaded') else 'Not loaded'}")
        if status.get('model_type'):
            st.info(f"Type: {status['model_type']}")
    else:
        st.error("Backend: Offline")
        st.warning("Make sure the Flask server is running on port 5001")

    st.markdown("---")
    st.header("Arrhythmia Classes")
    for cls, info in CLASS_DESCRIPTIONS.items():
        st.markdown(
            f"<div style='border-left: 4px solid {info['color']}; padding-left: 8px; margin-bottom: 8px;'>"
            f"<b>{cls}</b> — {info['severity']}<br>"
            f"<small>{info['desc']}</small></div>",
            unsafe_allow_html=True
        )

tab1, tab2, tab3 = st.tabs(["Upload ECG File", "Generate Sample", "Batch Analysis"])

with tab1:
    st.subheader("Upload ECG Signal CSV")
    st.markdown(
        "Upload a CSV file containing ECG amplitude values. "
        "Each row (or column) should contain one signal value. "
        "The model uses the first 187 samples."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='upload_tab')

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            signal_values = df.values.flatten().tolist()
            st.success(f"File loaded: {len(signal_values)} samples")

            display_signal = signal_values[:187] if len(signal_values) >= 187 else signal_values
            fig = plot_ecg_signal(display_signal, "Uploaded ECG Signal (first 187 samples)")
            st.pyplot(fig)
            plt.close(fig)

            if st.button("Classify ECG Signal", type="primary", key='classify_upload'):
                with st.spinner("Analyzing ECG signal..."):
                    uploaded_file.seek(0)
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            files={'file': ('ecg.csv', uploaded_file, 'text/csv')},
                            timeout=30
                        )
                        result = response.json()

                        if 'error' in result:
                            st.error(f"Prediction error: {result['error']}")
                        else:
                            pred = result['prediction']
                            conf = result['confidence']
                            info = CLASS_DESCRIPTIONS.get(pred, {})

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(
                                    f"<div style='background:{info.get('color','#aaa')}22; "
                                    f"border: 2px solid {info.get('color','#aaa')}; "
                                    f"border-radius:10px; padding:20px; text-align:center;'>"
                                    f"<h2 style='color:{info.get('color','#333')};'>{pred}</h2>"
                                    f"<p style='font-size:16px;'>{info.get('severity','')}</p>"
                                    f"<p><b>Confidence: {conf*100:.1f}%</b></p>"
                                    f"<p><small>{info.get('desc','')}</small></p>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                            with col2:
                                if 'all_probabilities' in result:
                                    fig2 = plot_probabilities(result['all_probabilities'])
                                    st.pyplot(fig2)
                                    plt.close(fig2)

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Make sure Flask server is running.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab2:
    st.subheader("Generate Sample ECG Signal")
    st.markdown("Generate a synthetic ECG signal for testing the classifier.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sample_class = st.selectbox("Select arrhythmia class", list(CLASS_DESCRIPTIONS.keys()))
    with col2:
        n_samples = st.slider("Signal length (samples)", 100, 500, 187)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Signal", type="primary")

    if generate_btn or 'sample_signal' in st.session_state:
        if generate_btn:
            signal = generate_sample_ecg(sample_class, n_samples)
            st.session_state['sample_signal'] = signal
            st.session_state['sample_class'] = sample_class

        signal = st.session_state['sample_signal']
        sample_class_display = st.session_state.get('sample_class', sample_class)

        fig = plot_ecg_signal(signal, f"Synthetic {sample_class_display} ECG Signal", sample_class_display)
        st.pyplot(fig)
        plt.close(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Classify This Signal", type="primary", key='classify_sample'):
                with st.spinner("Classifying..."):
                    try:
                        payload = {'signal': signal.tolist()}
                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            json=payload,
                            timeout=30
                        )
                        result = response.json()

                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            pred = result['prediction']
                            conf = result['confidence']
                            info = CLASS_DESCRIPTIONS.get(pred, {})

                            st.markdown(
                                f"<div style='background:{info.get('color','#aaa')}22; "
                                f"border: 2px solid {info.get('color','#aaa')}; "
                                f"border-radius:10px; padding:15px;'>"
                                f"<b>Prediction:</b> <span style='color:{info.get('color','#333')};font-size:18px;'>{pred}</span> "
                                f"— {info.get('severity','')} | "
                                f"<b>Confidence:</b> {conf*100:.1f}%"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            if 'all_probabilities' in result:
                                fig2 = plot_probabilities(result['all_probabilities'])
                                st.pyplot(fig2)
                                plt.close(fig2)
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col_b:
            csv_data = pd.DataFrame(signal, columns=['amplitude'])
            csv_bytes = csv_data.to_csv(index=False, header=False).encode()
            st.download_button(
                "Download as CSV",
                data=csv_bytes,
                file_name=f"ecg_{sample_class_display.lower()}_sample.csv",
                mime='text/csv'
            )

with tab3:
    st.subheader("Batch Analysis")
    st.markdown("Analyze multiple ECG windows at once by uploading a multi-row CSV (one ECG window per row).")

    batch_file = st.file_uploader("Upload multi-row CSV", type=['csv'], key='batch_tab')

    if batch_file is not None:
        try:
            df_batch = pd.read_csv(batch_file, header=None)
            st.info(f"Loaded {len(df_batch)} rows x {df_batch.shape[1]} columns")

            if st.button("Run Batch Classification", type="primary"):
                results = []
                progress = st.progress(0)
                status_text = st.empty()

                for i, row in df_batch.iterrows():
                    signal_vals = row.dropna().tolist()
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            json={'signal': signal_vals},
                            timeout=15
                        )
                        r = response.json()
                        results.append({
                            'Row': i + 1,
                            'Prediction': r.get('prediction', 'Error'),
                            'Confidence': f"{r.get('confidence', 0)*100:.1f}%",
                            'Signal Length': len(signal_vals)
                        })
                    except Exception as e:
                        results.append({
                            'Row': i + 1,
                            'Prediction': 'Error',
                            'Confidence': '0%',
                            'Signal Length': len(signal_vals)
                        })

                    progress.progress((i + 1) / len(df_batch))
                    status_text.text(f"Processing row {i+1}/{len(df_batch)}...")

                status_text.empty()
                progress.empty()

                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                summary = results_df['Prediction'].value_counts()
                st.markdown("**Summary:**")
                st.bar_chart(summary)

                csv_out = results_df.to_csv(index=False).encode()
                st.download_button("Download Results CSV", csv_out, "batch_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error reading batch file: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "ECG Arrhythmia Classification System | LSTM Deep Learning Model | "
    "For research/educational use only — not a medical device"
    "</div>",
    unsafe_allow_html=True
)
