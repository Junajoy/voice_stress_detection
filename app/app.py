
# app.py

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

st.set_page_config(page_title="Voice Stress Detection", layout="wide")
st.title("Voice Stress Detection App")
st.caption("Upload or record audio to analyze stress level")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    # Update the path to your local models folder
    return tf.keras.models.load_model(
        "models/cremad_cnn_speakerind.keras",
        compile=False
    )

model = load_model()
classes = ["Low Stress", "Medium Stress", "High Stress"]

# ---------------- FEATURE EXTRACTION ----------------
def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 130:
        mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :130]
    return mfcc[np.newaxis, ..., np.newaxis]

def extract_signal_features(y, sr):
    pitch = librosa.yin(y, fmin=50, fmax=300)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "Pitch Variability": float(np.std(pitch)) if len(pitch) > 0 else 0,
        "Spectral Centroid": float(np.mean(centroid)) if len(centroid) > 0 else 0,
        "MFCC Variance": float(np.var(mfcc)) if mfcc.size > 0 else 0
    }

# ---------------- STRESS OVER TIME ----------------
def stress_over_time(y, sr, window=2.0, hop=1.0):
    win = int(window * sr)
    step = int(hop * sr)
    if len(y) < win:
        return None
    seq = []
    for i in range(0, len(y) - win, step):
        seg = y[i:i + win]
        probs = model.predict(extract_mfcc(seg, sr), verbose=0)[0]
        idx = np.argmax(probs)
        # Map low/medium/high to stress intensity values
        if idx == 0:  # Low
            stress_value = 0.1 + 0.2 * probs[idx]
        elif idx == 1:  # Medium
            stress_value = 0.4 + 0.3 * probs[idx]
        else:  # High
            stress_value = 0.7 + 0.3 * probs[idx]
        seq.append(stress_value)
    return np.array(seq)

def smooth_curve(seq, alpha=0.6):
    if seq is None or len(seq) == 0:
        return seq
    smoothed = [seq[0]]
    for val in seq[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

# ---------------- PLOTS ----------------
def plot_waveform(y, stress_seq=None):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(y, color="blue")
    ax.set_facecolor("white")
    ax.set_title("Waveform", fontsize=10)
    HIGH_THRESHOLD = 0.7
    if stress_seq is not None and len(stress_seq) > 0:
        seg_len = len(y) // len(stress_seq)
        for i, val in enumerate(stress_seq):
            if val > HIGH_THRESHOLD:
                ax.axvspan(i*seg_len, (i+1)*seg_len, color="red", alpha=0.15)
    st.pyplot(fig)
    plt.close(fig)

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(4, 2))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.set_title("Spectrogram", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

def plot_stress_curve(seq):
    if seq is None or len(seq) == 0:
        st.info("Audio too short for stress-over-time analysis.")
        return
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(seq, color="red")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Stress Intensity")
    ax.set_xlabel("Time Segments")
    ax.set_title("Stress Over Time", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

# ---------------- AUDIO RECORDER ----------------
class Recorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame):
        self.frames.append(frame.to_ndarray())
        return frame

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------- UPLOADER & RECORD BUTTON ----------------
c1, c2 = st.columns([8, 1])

with c1:
    uploaded_files = st.file_uploader(
        "Select WAV/MP3 audio files",
        type=["wav", "mp3"],
        accept_multiple_files=True
    )

    if st.button("Show Stress Level Comparison"):
        if st.session_state.results:
            df_comp = pd.DataFrame(st.session_state.results).drop_duplicates(subset="File", keep="first").set_index("File")
            st.bar_chart(df_comp["Stress Score"])
        else:
            st.info("No audio processed yet for comparison.")

with c2:
    record_clicked = st.button("🎙️")

recorded_audio = None
if record_clicked:
    ctx = webrtc_streamer(
        key="rec",
        audio_processor_factory=Recorder,
        media_stream_constraints={"audio": True, "video": False},
    )
    if ctx.audio_processor and st.button("Stop Recording"):
        audio = np.concatenate(ctx.audio_processor.frames, axis=1)
        recorded_audio = audio.mean(axis=0)

# ---------------- MAIN PROCESS ----------------
def process_audio(y, sr, name):
    probs = model.predict(extract_mfcc(y, sr), verbose=0)[0]
    idx = np.argmax(probs)
    predicted_class = classes[idx]
    conf = float(probs[idx])

    stress_seq = stress_over_time(y, sr)
    stress_seq = smooth_curve(stress_seq)
    features = extract_signal_features(y, sr)

    st.subheader(name)
    st.metric(
        "Predicted Stress Level",
        predicted_class,
        f"{conf*100:.1f}% confidence"
    )
    st.progress(conf)

    left, right = st.columns(2)
    with left:
        plot_waveform(y, stress_seq)
        plot_stress_curve(stress_seq)
    with right:
        plot_spectrogram(y, sr)
        st.markdown("### Feature Indicators")
        st.bar_chart(features)

    st.markdown("### Actionable Insights")
    if predicted_class == "High Stress" and features["Pitch Variability"] > 0.15:
        st.warning("High pitch fluctuation detected — possible anxiety or nervousness.")
    if predicted_class == "High Stress":
        st.error("High stress detected. Consider rest or breathing exercises.")
    elif predicted_class == "Medium Stress":
        if features["Pitch Variability"] > 0.1:
            st.info("Moderate pitch fluctuation detected — some stress may be present.")
    else:
        st.info("Stress prediction indicates Low Stress — no alerts.")

    st.session_state.results.insert(0, {
        "File": name,
        "Stress Level": predicted_class,
        "Stress Score": float(np.mean(stress_seq)) if stress_seq is not None else conf,
        "Confidence": conf
    })

# ---------------- PROCESS FILES ----------------
if uploaded_files:
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.getbuffer())
        y, sr = librosa.load(tmp.name, sr=None)
        process_audio(y, sr, f.name)

if recorded_audio is not None:
    process_audio(recorded_audio, 16000, "Recorded Audio")

# ---------------- EXPORT RESULTS ----------------
if st.session_state.results:
    st.markdown("### Export Results")
    df = pd.DataFrame(st.session_state.results)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "stress_results.csv",
        "text/csv"
    )
    st.download_button(
        "Download JSON",
        df.to_json(orient="records"),
        "stress_results.json",
        "application/json"
    )
