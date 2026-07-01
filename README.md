# 🐦 KooKoo AI — Bird Sound Classification

A deep learning web app that identifies bird species from audio recordings. Upload a bird call (MP3, WAV, OGG, or FLAC), and a trained CNN model predicts the species from 114 possible classes, along with confidence scores, top-5 predictions, and audio visualizations (waveform + MFCC heatmap).

**Live demo:** _add your deployed Streamlit Cloud URL here after deploying_

![Main Interface](screenshots/main_interface.png)

## Features

- 🎯 Bird species classification across 114 species
- 📊 Top-5 predictions with confidence scores
- 🌊 Waveform and MFCC feature visualizations
- 💾 Downloadable prediction results (CSV)
- 🎧 In-browser audio playback

## How It Works

1. Audio is loaded with `librosa` at its native sample rate.
2. 40 MFCC (Mel-Frequency Cepstral Coefficients) are extracted and averaged across time.
3. The feature vector is fed into a trained CNN (`bird_model.h5`, Keras/TensorFlow).
4. The model outputs a probability distribution over 114 bird species, mapped via `prediction.json`.

## Tech Stack

- **Frontend/App:** [Streamlit](https://streamlit.io/)
- **ML:** TensorFlow / Keras (CNN)
- **Audio processing:** librosa, soundfile
- **Visualization:** Plotly, Pandas

## Running Locally

### Prerequisites
- Python 3.11 (recommended — TensorFlow does not yet support 3.13+)
- ~500 MB free disk space for dependencies

### Setup

```bash
git clone https://github.com/meenaprajapat/Bird-Sound-Classification.git
cd Bird-Sound-Classification

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
├── app.py                 # Streamlit application
├── bird_model.h5           # Trained CNN model
├── prediction.json         # Class index -> species name mapping
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages needed on Streamlit Cloud (libsndfile)
├── .streamlit/config.toml  # Streamlit theme & server config
└── screenshots/             # App screenshots for documentation
```

## Deployment (Streamlit Community Cloud — Free)

This repo is ready to deploy as-is:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and the system packages in `packages.txt` automatically.
5. First load may take a minute while the model loads — subsequent loads are cached (`@st.cache_resource`).

Your app will be live at `https://<your-app-name>.streamlit.app`.

## Model Notes

- Input: 40 MFCC coefficients (mean-aggregated across time), reshaped to `(1, 40, 1)`.
- Output: 114-way softmax over bird species.
- Reported test accuracy: ~65%.

## Author

**Meena Prajapat**
[GitHub](https://github.com/meenaprajapat) · [LinkedIn](https://www.linkedin.com/in/meena-prajapat-a166b4200/)
