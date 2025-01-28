import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import pickle
import queue
import time
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import io
import tempfile


# Load trained Random Forest Model
@st.cache_resource
def load_trained_model():
    try:
        with open("Random_forrest.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error(" Error: 'Random_forrest.pkl' not found. Please train and save the model first.")
        return None


rf_model = load_trained_model()

# Streamlit UI Setup
st.title("Real-Time Cough Detection")

# Mode selection: Live or Manual
mode = st.radio("Select Mode:", ["Live Monitoring", "Manual Recording"])

# Audio Settings
SAMPLE_RATE = 16000
FRAME_LENGTH = int(SAMPLE_RATE * 5)  # Process audio in 5-seconds chunks

# Streamlit placeholders for live updates
classification_placeholder = st.empty()
waveform_placeholder = st.empty()
mfcc_placeholder = st.empty()
history_placeholder = st.empty()
cough_graph_placeholder = st.empty()
alert_placeholder = st.empty()

# Ensure session storage for saved recordings and cough history
if "recordings" not in st.session_state:
    st.session_state["recordings"] = {}

if "cough_history" not in st.session_state:
    st.session_state["cough_history"] = []

if "cough_count" not in st.session_state:
    st.session_state["cough_count"] = []


# Function to extract MFCCs
def extract_mfccs(audio_chunk, sr=SAMPLE_RATE):
    if audio_chunk.ndim > 1:
        audio_chunk = np.mean(audio_chunk, axis=1)  # Convert stereo to mono
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)


# Function to detect cough based on loudness threshold
def detect_cough(audio_chunk):
    rms_energy = np.sqrt(np.mean(audio_chunk ** 2))  # Root Mean Square Energy
    return rms_energy > 0.005  # Lower threshold to detect softer coughs


### ** Live Monitoring Mode**
if mode == "Live Monitoring":
    monitoring = st.toggle("Start Live Monitoring")

    if monitoring:
        st.write("Listening... Speak into the microphone.")

        while monitoring:
            audio_chunk = sd.rec(FRAME_LENGTH, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()  # Wait for the recording to complete
            audio_chunk = audio_chunk.flatten()

            # Detect cough event
            if detect_cough(audio_chunk):
                mfcc_features = extract_mfccs(audio_chunk).reshape(1, -1)
                prediction = rf_model.predict(mfcc_features)[0]
                confidence = rf_model.predict_proba(mfcc_features)[0][prediction] * 100

                label_text = "✅ Cough Detected" if prediction == 1 else "❌ No Cough"

                # Store event in history
                detected_event = {
                    "Time": time.strftime("%H:%M:%S"),
                    "Prediction": label_text,
                    "Confidence": confidence
                }
                st.session_state["cough_history"].append(detected_event)
                st.session_state["cough_count"].append(time.time())

                # Live Update UI
                classification_placeholder.subheader(f"Prediction: {label_text}")
                classification_placeholder.text(f"Confidence: {confidence:.2f}%")

                # Live Plot Waveform
                fig, ax = plt.subplots(figsize=(6, 2))
                librosa.display.waveshow(audio_chunk, sr=SAMPLE_RATE, ax=ax)
                ax.set_title("Detected Cough Event - Live Waveform")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                waveform_placeholder.pyplot(fig)

                # Display MFCCs in Live Table
                mfcc_df = pd.DataFrame(mfcc_features, columns=[f"MFCC_{i + 1}" for i in range(13)])
                mfcc_placeholder.dataframe(mfcc_df)

                # Display Cough History
                history_placeholder.subheader("📜 Cough Detection History")
                history_placeholder.table(pd.DataFrame(st.session_state["cough_history"]))

                # Display Cough Trend Graph
                cough_graph_placeholder.subheader("📊 Cough Detection Trend")
                cough_graph_placeholder.line_chart(pd.DataFrame(st.session_state["cough_count"], columns=["Timestamp"]))

                # Alert when 10 coughs are detected
                if len(st.session_state["cough_history"]) >= 10:
                    alert_placeholder.error("🚨 WARNING: 10 cough events detected! Consider seeing a doctor.")

            time.sleep(1)  # Process every second

    else:
        st.write(" Monitoring Stopped.")

### ** Manual Recording Mode**
elif mode == "Manual Recording":
    duration = st.slider("Recording Duration (seconds)", 3, 5, 7)

    if st.button("🎙️ Start Recording"):
        st.write("Recording...")
        recorded_audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        st.write("Recording Complete.")

        #  Convert float32 → int16 for saving
        recorded_audio_int16 = (recorded_audio * 32767).astype(np.int16)

        # Save Audio using mkstemp() to prevent deletion issues
        temp_fd, temp_filename = tempfile.mkstemp(suffix=".wav")
        sf.write(temp_filename, recorded_audio_int16, SAMPLE_RATE)

        # Play recorded audio correctly
        st.audio(temp_filename, format="audio/wav")

        # Convert int16 back to float32 for Librosa processing
        y = recorded_audio_int16.astype(np.float32) / 32767

        # Ensure waveform is correctly displayed
        st.subheader("📈 Recorded Audio Waveform")
        fig, ax = plt.subplots(figsize=(6, 2))
        librosa.display.waveshow(y.flatten(), sr=SAMPLE_RATE, ax=ax)
        ax.set_title("Recorded Audio Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to avoid display issues

        # Extract MFCCs
        mfcc_features = extract_mfccs(y).reshape(1, -1)

        # Display MFCCs as numerical table
        st.subheader("📊 Extracted MFCC Features")
        mfcc_df = pd.DataFrame(mfcc_features, columns=[f"MFCC_{i+1}" for i in range(13)])
        st.dataframe(mfcc_df)

        # Fix: Classify the recorded audio using trained model
        if rf_model:
            prediction = rf_model.predict(mfcc_features)[0]
            confidence = rf_model.predict_proba(mfcc_features)[0][prediction] * 100
            label_text = "✅ Cough Detected" if prediction == 1 else "❌ No Cough"

            # Display results
            st.subheader(f"Prediction: {label_text}")
            st.text(f"Confidence: {confidence:.2f}%")
        else:
            st.error(" No trained model found. Please train and save the model first.")





