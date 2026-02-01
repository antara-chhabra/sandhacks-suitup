"""
Voice Emotion Analyzer GUI Application
Usage: python app.py
"""
import os
import sys
import json
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
import whisper
import librosa
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import ttk
import joblib

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_audio_features

# =========================
# Settings
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
JSON_PATH = os.path.join(BASE_DIR, "voice_logs.json")
MODEL_PATH = os.path.join(BASE_DIR, "clf.pkl")

# Audio settings
FS = 44100
CHANNELS = 1

# Vocal states (must match training)
VOCAL_STATES = ["nervous", "confident", "overconfident", "monotone"]

# Global variables
model = None
clf = None

# =========================
# UI Helpers
# =========================
def safe_update_label(text):
    root.after(0, lambda: status_label.config(text=text))

def safe_insert_text(text):
    root.after(0, lambda: transcription_box.insert(tk.END, text))

def safe_insert_features(text):
    root.after(0, lambda: feature_box.insert(tk.END, text + "\n"))


# =========================
# Model Loading
# =========================
def load_whisper_model():
    """Load Whisper model in a background thread."""
    global model
    safe_update_label("Loading Whisper model...")
    model = whisper.load_model("base")
    root.after(0, lambda: record_button.config(state="normal"))
    safe_update_label("Ready")


def load_ml_model():
    """Load trained ML classifier in a background thread."""
    global clf
    safe_update_label("Loading ML classifier...")
    
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        safe_update_label("ML classifier loaded.")
    else:
        safe_update_label("ML model not found. Run train_model.py first!")


# =========================
# ML Classification
# =========================
def classify_vocal_state(features):
    """Classify vocal state using trained model."""
    if clf is None:
        return "unknown"
    
    features_array = features.reshape(1, -1)
    pred = clf.predict(features_array)[0]
    return VOCAL_STATES[pred]


# =========================
# Save Results
# =========================
def save_result(transcription, features, state, audio_path):
    """Save analysis results to JSON."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "transcription": transcription,
        "features": features.tolist() if isinstance(features, np.ndarray) else features,
        "predicted_state": state,
        "audio_file": os.path.basename(audio_path)
    }
    
    data = []
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
    
    data.append(entry)
    
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


# =========================
# Recording & Processing
# =========================
def record_and_process():
    """Record audio and analyze it."""
    try:
        duration = int(duration_var.get())
        safe_update_label(f"Recording {duration}s...")
        
        # Record audio
        recording = sd.rec(int(duration * FS), samplerate=FS, channels=CHANNELS)
        sd.wait()
        
        # Normalize to int16
        recording_int16 = np.int16(recording / np.max(np.abs(recording)) * 32767)
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"recording_{timestamp}.wav"
        audio_path = os.path.join(RECORDINGS_DIR, audio_filename)
        
        write(audio_path, FS, recording_int16)
        
        # Transcribe with Whisper
        safe_update_label("Transcribing...")
        result = model.transcribe(audio_path)
        transcription = result["text"].strip()
        safe_insert_text(transcription + "\n")
        
        # Extract features and classify
        safe_update_label("Analyzing voice...")
        features = extract_audio_features(audio_path)
        state = classify_vocal_state(features)
        
        # Display features
        feature_text = (
            f"Pitch: {features[0]:.2f} | Std Pitch: {features[1]:.2f} | "
            f"Volume: {features[2]:.4f} | Std Volume: {features[3]:.4f} | "
            f"Spectral Centroid: {features[4]:.2f} | Emotion: {state}"
        )
        safe_insert_features(feature_text)
        
        # Save results
        save_result(transcription, features, state, audio_path)
        
        safe_update_label(f"Detected: {state}")
        
    except Exception as e:
        safe_update_label(f"Error: {e}")


def start_recording_thread():
    """Start recording in a separate thread."""
    threading.Thread(target=record_and_process, daemon=True).start()


# =========================
# UI Setup
# =========================
# Create recordings directory
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Create main window
root = tk.Tk()
root.title("Voice Emotion Analyzer (ML)")
root.geometry("700x550")

# Main frame
frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

# Duration selector
ttk.Label(frame, text="Recording duration (seconds)").pack(anchor="w")

duration_var = tk.StringVar(value="5")
ttk.Combobox(frame, textvariable=duration_var,
             values=["3", "5", "10", "15"],
             state="readonly").pack(anchor="w")

# Record button
record_button = ttk.Button(frame,
                           text="Record Voice",
                           state="disabled",
                           command=start_recording_thread)
record_button.pack(pady=10)

# Status label
status_label = ttk.Label(frame, text="Initializing...")
status_label.pack()

# Transcription section
transcription_frame = ttk.LabelFrame(frame, text="Transcription")
transcription_frame.pack(fill="x", pady=5)

transcription_box = tk.Text(transcription_frame, height=6)
transcription_box.pack(fill="x")

# Features section
feature_frame = ttk.LabelFrame(frame, text="Vocal Features")
feature_frame.pack(fill="x", pady=5)

feature_box = tk.Text(feature_frame, height=6)
feature_box.pack(fill="x")

# =========================
# Start Application
# =========================
if __name__ == "__main__":
    # Load models in background threads
    threading.Thread(target=load_whisper_model, daemon=True).start()
    threading.Thread(target=load_ml_model, daemon=True).start()
    
    # Run the GUI
    root.mainloop()

