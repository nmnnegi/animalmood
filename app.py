import streamlit as st
import numpy as np
import sounddevice as sd
from animal_mood_detector import analyze_live_audio

# Streamlit UI
st.title("🐾 Animal Mood Detector 🎵")

# Select an animal
animal = st.selectbox("Choose an Animal:", ["Dog", "Cat", "Bird", "Lion", "Snake"])

# Set audio parameters
fs = 22050  # Sampling rate
duration = 3  # Recording duration

if st.button("🎙️ Start Live Detection"):
    st.write("🎤 Recording...")

    # Record live audio
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    
    st.write("✅ Audio Captured! Analyzing...")

    # Analyze the live audio
    emotion = analyze_live_audio(audio, fs, animal)

    # Display result
    st.subheader("🎭 Detected Emotion")
    st.success(emotion)
