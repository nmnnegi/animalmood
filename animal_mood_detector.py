import numpy as np
import tensorflow_hub as hub
import librosa

# Load YAMNet model
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Define emotion mapping for different animals
SOUND_TO_EMOTION = {
    "Bark": "Alert",
    "Meow": "Calm",
    "Growl": "Aggressive",
    "Whimper": "Sad",
    "Bird chirp": "Happy",
    "Roar": "Angry",
    "Hiss": "Defensive",
}

def analyze_live_audio(audio, fs, animal):
    """
    Process live audio with YAMNet and return an emotion.
    
    Args:
    - audio: Raw audio data (NumPy array)
    - fs: Sampling rate
    - animal: Selected animal
    
    Returns:
    - Detected emotion
    """
    # Flatten and resample audio for YAMNet (16kHz required)
    audio = audio.flatten()
    audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)

    # Run through YAMNet model
    scores, embeddings, spectrogram = model(audio)

    # Get the highest confidence sound label
    sound_label = scores.numpy().argmax()

    # Map sound to an emotion
    detected_emotion = SOUND_TO_EMOTION.get(sound_label, "Neutral")

    return detected_emotion
