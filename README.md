# Cough_Audio_Detection_ML
A web app that detects cough audio and displays the classification results in Streamlit. 

## Project Overview
This project is a real-time cough detection system built using Python, Streamlit, and Machine Learning. It records audio via a microphone, extracts MFCC features, and classifies the sound as cough or non-cough using a Random Forest classifier.

### Features
- Real-time & Manual Recording – Users can record live audio or manually upload files.
- Feature Extraction – Extracts MFCCs & Spectrograms from audio.
- Machine Learning Classification – Detects cough events using Random Forest Classifier.
- Data Visualization – Displays waveform, MFCCs, and classification results.
- Live Monitoring – Continuously detects coughs and logs them.
- Cough History & Alerts – Tracks cough trends and alerts users when multiple coughs are detected.

---

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Cough_Detection_AI.git
cd Cough_Detection_AI
```

### 2. Install Dependencies
Create a virtual environment (recommended) and install required packages:
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## Usage Guide
### 1. Running the App
After launching the app, select a mode:
- Live Monitoring: Real-time audio detection.
- Manual Recording: Record & analyze short audio clips.

### 2. Uploading and Classifying Audio
- Click "Upload File" to test existing `.wav` files.
- View waveform, MFCCs, and spectrogram.
- Model will classify audio as Cough or Non-Cough.

### 3. Training the Model
- Upload cough and non-cough audio samples.
- Extracted MFCCs are used to train a Random Forest model.
- Model is saved as `Random_forrest.pkl` for future use.

### 4. Real-time Cough Monitoring
- Click "Start Live Monitoring" to continuously detect coughs.
- If 10 coughs are detected within a short period, an alert is displayed.
- View cough event history & trend graphs.

---

## Project Structure
```
Cough_Detection_AI/
│── app.py                 # Main Streamlit App
│── Random_forrest.pkl     # Pre-trained ML Model
│── requirements.txt       # Dependencies
│── data/
│   ├── cough_1.wav       # Example cough audio
│   ├── speech_1.wav      # Example non-cough audio
│── utils/
│   ├── feature_extraction.py  # MFCC & Spectrogram Functions
│   ├── audio_processing.py    # Audio Preprocessing
│   ├── model_training.py      # Machine Learning Model Training
```

---

## Technologies Used
- Programming: Python 3.13.1
- Libraries: `Streamlit`, `SoundDevice`, `Librosa`, `Scikit-Learn`, `Pandas`, `Matplotlib`
- Machine Learning: Random Forest Classifier

---

## Future Improvements
Planned Features:
- IoT Integration (e.g., use with smartwatches, medical devices).
- Cloud Deployment (Google Cloud, AWS, Azure).
- CNN Deep Learning Model for improved accuracy.
- Mobile App Version (Android/iOS).

---

## Contact
Email: [akaqinlang@gmail.com](mailto:akaqinlang@gmail.com)
GitHub: [Scottqin2001404](https://github.com/Scottqin2001404)
