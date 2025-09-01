import streamlit as st
import librosa, numpy as np, tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
import tempfile

# Load model
model = tf.keras.models.load_model("cnn_tess.h5")
emotions = {
    "angry": "😡 Angry",
    "disgust": "🤢 Disgust",
    "fear": "😨 Fear",
    "happy": "😄 Happy",
    "neutral": "😐 Neutral",
    "sad": "😢 Sad",
    "surprise": "😲 Surprise"
}

# --- Page config ---
st.set_page_config(page_title="Speech Emotion Recognition 🎙️", page_icon="🎵", layout="centered")

# --- Sidebar ---
st.sidebar.header("ℹ️ How to use")
st.sidebar.write("""
1. Upload a **.wav audio file**.  
2. Wait a few seconds for processing.  
3. The app will predict your **emotion** 🎯  
""")

# --- Main title ---
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file and let the AI guess the **emotion**!")

# --- Upload input ---
audio_file = st.file_uploader("📂 Upload your audio file (.wav)", type=["wav"])

if audio_file is not None:
    # Save audio to temp file so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getbuffer())
        tmp_path = tmp_file.name

    # Load audio
    y, sr = librosa.load(tmp_path, sr=16000)

    # Show waveform
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="steelblue")
    ax.set(title="Your Voice Waveform")
    st.pyplot(fig)

    # --- Feature extraction (log-mel spectrogram) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix shape: transpose so it's (time, features)
    mel_db = mel_db.T

    # Pad/trim to match model input (130 time frames)
    max_len = 130
    if mel_db.shape[0] < max_len:
        pad_width = max_len - mel_db.shape[0]
        mel_db = np.pad(mel_db, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mel_db = mel_db[:max_len, :]

    # Final shape: (1, 130, 128, 1)
    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.expand_dims(mel_db, axis=0)

    # --- Predict ---
    pred = model.predict(mel_db)
    predicted_emotion = list(emotions.keys())[np.argmax(pred)]

    # Display result in a nice card
    st.markdown(
        f"""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:15px; text-align:center;">
            <h2 style="color:#333;">Prediction 🎯</h2>
            <h1 style="color:#4CAF50;">{emotions[predicted_emotion]}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show probability bar chart
    st.subheader("Prediction Confidence")
    st.bar_chart({emotions[k]: float(v) for k, v in zip(emotions.keys(), pred[0])})
    st.write("Note: The model may not be 100% accurate. Try different audio samples!")


