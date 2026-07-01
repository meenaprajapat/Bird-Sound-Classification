import streamlit as st
import librosa
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pandas as pd
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="🐦 KooKoo AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --grad: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --grad-soft: linear-gradient(135deg, #7f7fd5 0%, #86a8e7 50%, #91eae4 100%);
        --glass-bg: rgba(255, 255, 255, 0.55);
        --glass-border: rgba(255, 255, 255, 0.6);
    }

    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Animated aurora background */
    .stApp {
        background: linear-gradient(-45deg, #e0eafc, #cfdef3, #e5d4f1, #d3e9f7);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .block-container {
        padding: 3rem 3rem 2rem 3rem;
        max-width: 1400px;
    }

    /* Give the top header breathing room so it never gets clipped */
    .main-header {
        margin-top: 0.5rem;
    }

    /* Make horizontal card rows equal height */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    /* Force every nested wrapper inside a column to fill full height
       so the card stretches to match the tallest one in the row */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stMarkdownContainer"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stMarkdownContainer"] > div,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .element-container {
        height: 100%;
    }
    .feature-card,
    .metric-card {
        height: 100%;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* "How It Works" step cards (inline-styled) — force equal size */
    .step-card {
        height: 100%;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 1.5rem 1rem;
        background: var(--glass-bg);
        backdrop-filter: blur(14px) saturate(150%);
        -webkit-backdrop-filter: blur(14px) saturate(150%);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        box-shadow: 0 8px 28px rgba(31, 38, 135, 0.14);
        transition: all 0.3s ease;
    }
    .step-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 45px rgba(102, 126, 234, 0.28);
    }

    /* Shared glass surface */
    .main-header,
    .upload-section,
    .metric-card,
    .feature-card,
    .footer,
    .info-box,
    .success-box {
        background: var(--glass-bg);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }

    /* Header Styling */
    .main-header {
        position: relative;
        overflow: hidden;
        background: linear-gradient(-45deg, #667eea, #764ba2, #6a3fb5, #5b73e8);
        background-size: 300% 300%;
        animation: headerGradient 12s ease infinite, fadeInDown 0.8s ease-out;
        border: 1px solid rgba(255,255,255,0.25);
        padding: 3.2rem 2rem;
        border-radius: 28px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 18px 50px rgba(102, 126, 234, 0.45);
    }

    @keyframes headerGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Decorative glowing orbs */
    .main-header::before,
    .main-header::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        filter: blur(2px);
        opacity: 0.35;
    }
    .main-header::before {
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(145,234,228,0.8) 0%, transparent 70%);
        top: -80px; left: -60px;
    }
    .main-header::after {
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(240,147,251,0.7) 0%, transparent 70%);
        bottom: -110px; right: -70px;
    }

    .main-header h1 {
        position: relative;
        z-index: 1;
        color: white;
        font-size: 3.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
        text-shadow: 0 4px 18px rgba(0,0,0,0.25);
        animation: floatTitle 4s ease-in-out infinite;
    }

    @keyframes floatTitle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }

    .main-header .subtitle {
        position: relative;
        z-index: 1;
        color: rgba(255,255,255,0.92);
        font-size: 1.25rem;
        font-weight: 300;
        margin-top: 0.6rem;
        letter-spacing: 0.3px;
    }

    .header-badge {
        position: relative;
        z-index: 1;
        display: inline-block;
        margin-top: 1.2rem;
        padding: 0.45rem 1.3rem;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 50px;
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: var(--grad);
        color: white;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Upload Section */
    .upload-section {
        padding: 3rem;
        border-radius: 24px;
        margin: 2rem 0;
        border: 2px dashed rgba(102, 126, 234, 0.7);
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #764ba2;
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.25);
        transform: translateY(-2px);
    }

    .upload-text {
        text-align: center;
        padding: 2rem;
    }

    .upload-text h2 {
        color: #5a3fa0;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(240,147,251,0.92) 0%, rgba(245,87,108,0.92) 100%);
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 3rem;
        border-radius: 28px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 18px 55px rgba(245, 87, 108, 0.35);
        animation: slideInUp 0.6s ease-out;
    }

    .prediction-box h2 {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .prediction-box h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .prediction-box h3 {
        color: rgba(255,255,255,0.95);
        font-size: 1.5rem;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-card {
        padding: 2rem;
        border-radius: 22px;
        border-top: 4px solid #667eea;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }

    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 45px rgba(102, 126, 234, 0.28);
    }

    .metric-card h4 {
        color: #5a3fa0;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    /* Info Boxes */
    .info-box {
        background: rgba(224, 247, 250, 0.6);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #00bcd4;
        margin: 1rem 0;
    }

    .info-box h3 {
        color: #00838f;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .success-box {
        background: rgba(200, 230, 201, 0.6);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4c3f8f 0%, #5b4b9e 45%, #3f5aa6 100%);
        border-right: 1px solid rgba(255,255,255,0.12);
        box-shadow: 4px 0 24px rgba(0,0,0,0.15);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] .element-container {
        color: rgba(255,255,255,0.95);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: rgba(255,255,255,0.88) !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Glass info panels inside the sidebar */
    section[data-testid="stSidebar"] div[style*="rgba(255,255,255,0.1)"] {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 14px !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    /* Sidebar divider lines */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18) !important;
    }

    /* --- Compact sidebar layout --- */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }

    .sb-brand {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .sb-logo {
        font-size: 3rem;
        line-height: 1;
        filter: drop-shadow(0 4px 10px rgba(0,0,0,0.25));
    }
    .sb-brand-name {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.4rem;
        letter-spacing: 0.5px;
    }
    .sb-brand-tag {
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    .sb-section {
        margin-bottom: 1.1rem;
    }
    .sb-heading {
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin: 0 0 0.5rem 0 !important;
    }
    .sb-panel {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .sb-panel p {
        margin: 0.25rem 0 !important;
        font-size: 0.88rem !important;
        color: rgba(255,255,255,0.9) !important;
    }
    .sb-panel li {
        margin: 0.3rem 0;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.9);
    }

    /* Format chips */
    .sb-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .sb-chip {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 50px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.5px;
    }

    .sb-footer {
        text-align: center;
        color: rgba(255,255,255,0.65);
        font-size: 0.78rem;
        line-height: 1.6;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.15);
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* File Uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 6px 22px rgba(31, 38, 135, 0.12);
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: var(--grad);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.5);
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: var(--grad);
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        border-radius: 22px;
        margin-top: 3rem;
    }

    .footer p {
        color: #5a3fa0;
        font-weight: 600;
    }

    /* Audio Player */
    .stAudio {
        border-radius: 15px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.6);
        border-radius: 12px;
        font-weight: 600;
        color: #5a3fa0;
    }

    /* Feature Cards */
    .feature-card {
        padding: 2rem;
        border-radius: 22px;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }

    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 24px 55px rgba(102, 126, 234, 0.3);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        color: #5a3fa0;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        color: #555;
        font-size: 1rem;
    }

    /* Stats Display */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }

    .stat-item {
        text-align: center;
        padding: 1.5rem;
    }

    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        background: var(--grad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        color: #555;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Responsive: mobile / small screens */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem;
        }
        .main-header {
            padding: 1.5rem 1rem;
        }
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header .subtitle {
            font-size: 0.95rem;
        }
        .header-badge {
            font-size: 0.72rem;
            padding: 0.4rem 0.9rem;
        }
        .upload-section {
            padding: 1.5rem;
        }
        .upload-text h2 {
            font-size: 1.4rem;
        }
        .prediction-box {
            padding: 1.5rem;
        }
        .prediction-box h1 {
            font-size: 1.8rem;
        }
        .prediction-box h2 {
            font-size: 1.3rem;
        }
        .metric-card,
        .feature-card {
            padding: 1.2rem;
        }
        .metric-card h2 {
            font-size: 1.8rem;
        }
        .feature-icon {
            font-size: 2rem;
        }
        .stat-number {
            font-size: 2rem;
        }
        .stButton>button {
            font-size: 1rem;
            padding: 0.8rem 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and prediction dictionary
@st.cache_resource
def load_model_and_dict():
    """Load the trained model and prediction dictionary"""
    try:
        model_path = Path("bird_model.h5")
        json_path = Path("prediction.json")

        if not model_path.exists():
            st.error("❌ bird_model.h5 not found in the current directory!")
            return None, None
        
        if not json_path.exists():
            st.error("❌ prediction.json not found in the current directory!")
            return None, None
        
        model = keras.models.load_model(str(model_path))
        
        with open(json_path, 'r') as f:
            prediction_dict = json.load(f)
        
        return model, prediction_dict
    except Exception as e:
        st.error(f"❌ Error loading model or dictionary: {e}")
        return None, None

def extract_mfcc_features(audio_file, n_mfcc=40):
    """Extract MFCC features from audio file - matches notebook exactly"""
    try:
        # Load audio WITHOUT forcing sample rate (matches notebook)
        # This uses the original sample rate of the audio file
        audio, sample_rate = librosa.load(audio_file)
        
        # Extract MFCC features (exactly as in notebook)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Aggregate using mean across time axis (exactly as in notebook)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Reshape for model input
        mfccs_mean = np.expand_dims(mfccs_mean, axis=0)  # Add batch dimension
        mfccs_mean = np.expand_dims(mfccs_mean, axis=2)  # Add channel dimension
        
        return mfccs_mean, audio, sample_rate, mfccs
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None, None, None, None

def predict_bird_species(audio_file, model, prediction_dict):
    """Predict bird species from audio file"""
    mfccs_features, audio, sr, mfccs_full = extract_mfcc_features(audio_file)
    
    if mfccs_features is None:
        return None, None, None, None, None
    
    # Make prediction
    prediction = model.predict(mfccs_features, verbose=0)
    
    # Get top prediction
    predicted_label = np.argmax(prediction)
    predicted_species = prediction_dict[str(predicted_label)]
    confidence = np.max(prediction) * 100
    
    # Get top 5 predictions
    top_5_indices = np.argsort(prediction[0])[-5:][::-1]
    top_5_predictions = []
    
    for idx in top_5_indices:
        species = prediction_dict[str(idx)]
        conf = prediction[0][idx] * 100
        top_5_predictions.append({'Species': species, 'Confidence': f'{conf:.2f}%', 'Score': conf})
    
    return predicted_species, confidence, top_5_predictions, audio, sr, mfccs_full

def plot_waveform(audio, sr):
    """Plot audio waveform using Plotly"""
    time = np.linspace(0, len(audio) / sr, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=audio,
        mode='lines',
        name='Waveform',
        line=dict(color='#4CAF50', width=1)
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=300,
        hovermode='x unified'
    )
    
    return fig

def plot_mfcc(mfccs, sr):
    """Plot MFCC features using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        x=np.arange(mfccs.shape[1]),
        y=np.arange(mfccs.shape[0]),
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='MFCC Features Heatmap',
        xaxis_title='Time Frames',
        yaxis_title='MFCC Coefficients',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_top5_predictions(top_5_predictions):
    """Plot top 5 predictions as a horizontal bar chart"""
    df = pd.DataFrame(top_5_predictions)
    
    fig = px.bar(
        df,
        x='Score',
        y='Species',
        orientation='h',
        text='Confidence',
        color='Score',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        title='Top 5 Predictions',
        xaxis_title='Confidence Score (%)',
        yaxis_title='Bird Species',
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

# Main application
def main():
    # Professional Header
    st.markdown("""
    <div class='main-header'>
        <h1>🎵 KooKoo AI 🐦</h1>
        <p class='subtitle'>Advanced Deep Learning System for Identifying Bird Species by Their Calls</p>
        <span class='header-badge'>🧠 CNN Model &nbsp;•&nbsp; 🎼 40 MFCC Features &nbsp;•&nbsp; 🐦 114 Species</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with Professional Design
    with st.sidebar:
        # Brand / Logo
        st.markdown("""
        <div class='sb-brand'>
            <div class='sb-logo'>🐦</div>
            <div class='sb-brand-name'>KooKoo AI</div>
            <div class='sb-brand-tag'>Bird Sound Classifier</div>
        </div>
        """, unsafe_allow_html=True)

        # About Section
        st.markdown("""
        <div class='sb-section'>
            <h3 class='sb-heading'>📚 About the Model</h3>
            <div class='sb-panel'>
                <p><strong>🧠 Architecture:</strong> Optimized CNN</p>
                <p><strong>🎼 Features:</strong> 40 MFCC Coefficients</p>
                <p><strong>🐦 Species:</strong> 114 Bird Types</p>
                <p><strong>⚡ Framework:</strong> TensorFlow 2.x</p>
                <p><strong>📊 Accuracy:</strong> ~65%+ on Test Set</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # How to Use
        st.markdown("""
        <div class='sb-section'>
            <h3 class='sb-heading'>🎯 Quick Guide</h3>
            <div class='sb-panel'>
                <ol style='padding-left: 1.1rem; margin: 0;'>
                    <li>📤 Upload bird sound file</li>
                    <li>🎧 Listen to preview</li>
                    <li>🚀 Click "Classify Species"</li>
                    <li>📊 View detailed results</li>
                    <li>💾 Download predictions</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Supported Formats
        st.markdown("""
        <div class='sb-section'>
            <h3 class='sb-heading'>📁 Supported Formats</h3>
            <div class='sb-chips'>
                <span class='sb-chip'>MP3</span>
                <span class='sb-chip'>WAV</span>
                <span class='sb-chip'>OGG</span>
                <span class='sb-chip'>FLAC</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Footer in Sidebar
        st.markdown("""
        <div class='sb-footer'>
            Made with ❤️ using <strong>TensorFlow &amp; Streamlit</strong><br>
            © 2025 KooKoo AI
        </div>
        """, unsafe_allow_html=True)
    
    # Load model and dictionary
    model, prediction_dict = load_model_and_dict()
    
    if model is None or prediction_dict is None:
        st.error("❌ Failed to load model or prediction dictionary. Please check if files exist.")
        return
    
    # Success Message
    st.markdown("""
    <div class='success-box'>
        <strong>✅ Model Loaded Successfully!</strong> The AI is ready to classify bird sounds from 114 different species.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 📤 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'ogg', 'flac'],
        help="Upload a bird sound audio file in MP3, WAV, OGG, or FLAC format"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎧 Audio Player")
            st.audio(uploaded_file, format='audio/mp3')
            
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.json(file_details)
        
        with col2:
            st.markdown("### 🔍 Analysis Options")
            
            show_waveform = st.checkbox("Show Waveform", value=True)
            show_mfcc = st.checkbox("Show MFCC Features", value=True)
            show_top5 = st.checkbox("Show Top 5 Predictions", value=True)
        
        # Prediction button
        st.markdown("---")
        if st.button("🎯 Classify Bird Species", type="primary"):
            with st.spinner('🔄 Analyzing audio and making predictions...'):
                # Save uploaded file to a temp path for librosa to read
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_audio_path = tmp.name

                try:
                    # Make prediction
                    predicted_species, confidence, top_5_predictions, audio, sr, mfccs_full = predict_bird_species(
                        temp_audio_path, model, prediction_dict
                    )
                finally:
                    os.remove(temp_audio_path)
                
                if predicted_species is not None:
                    # Display main prediction
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Main prediction box
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <h2>🐦 Predicted Species</h2>
                        <h1 style='font-size: 2.5rem; margin: 1rem 0;'>{predicted_species}</h1>
                        <h3>📊 Confidence: {confidence:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    st.markdown("### 📈 Prediction Metrics")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.markdown("""
                        <div class='metric-card'>
                            <h4>🎯 Top Prediction</h4>
                            <h2 style='color: #4CAF50;'>{:.2f}%</h2>
                        </div>
                        """.format(confidence), unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>📊 Total Classes</h4>
                            <h2 style='color: #2196F3;'>{len(prediction_dict)}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>⏱️ Audio Duration</h4>
                            <h2 style='color: #FF9800;'>{len(audio)/sr:.2f}s</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top 5 predictions
                    if show_top5:
                        st.markdown("---")
                        st.markdown("### 📊 Top 5 Predictions")
                        
                        col_chart, col_table = st.columns([2, 1])
                        
                        with col_chart:
                            fig_top5 = plot_top5_predictions(top_5_predictions)
                            st.plotly_chart(fig_top5, use_container_width=True)
                        
                        with col_table:
                            st.dataframe(
                                pd.DataFrame(top_5_predictions)[['Species', 'Confidence']],
                                hide_index=True,
                                use_container_width=True
                            )
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("### 📊 Audio Visualizations")
                    
                    if show_waveform:
                        st.markdown("#### 🌊 Waveform")
                        fig_waveform = plot_waveform(audio, sr)
                        st.plotly_chart(fig_waveform, use_container_width=True)
                    
                    if show_mfcc:
                        st.markdown("#### 🎼 MFCC Features")
                        fig_mfcc = plot_mfcc(mfccs_full, sr)
                        st.plotly_chart(fig_mfcc, use_container_width=True)
                        
                        with st.expander("ℹ️ What are MFCC Features?"):
                            st.markdown("""
                            **MFCC (Mel-Frequency Cepstral Coefficients)** are features extracted from audio signals that:
                            
                            - 🎵 Represent the short-term power spectrum of sound
                            - 👂 Mimic human auditory perception
                            - 📊 Capture frequency and temporal characteristics
                            - 🎯 Are widely used in audio classification tasks
                            
                            Each row represents a different MFCC coefficient, and each column represents a time frame.
                            """)
                    
                    # Download results
                    st.markdown("---")
                    st.markdown("### 💾 Download Results")
                    
                    results_df = pd.DataFrame(top_5_predictions)
                    csv = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("❌ Failed to make prediction. Please try another audio file.")
    
    else:
        # Professional Welcome Screen
        st.markdown("""
        <div class='upload-section'>
            <div style='text-align: center;'>
                <div style='font-size: 5rem; margin-bottom: 1rem;'>🎵</div>
                <h2 style='color: #667eea; font-size: 2.5rem; font-weight: 700;'>
                    Welcome to KooKoo AI Classification System
                </h2>
                <p style='font-size: 1.3rem; color: #666; margin: 1.5rem 0;'>
                    Upload a bird sound audio file to identify the species using our advanced AI model
                </p>
                <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 15px; margin-top: 2rem;'>
                    <p style='color: #667eea; font-size: 1.1rem; font-weight: 600;'>
                        📁 Supported Formats: MP3, WAV, OGG, FLAC
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Cards
        st.markdown("<br><h2 style='text-align: center; color: #667eea;'>✨ Key Features</h2>", unsafe_allow_html=True)
        
        feat_cols = st.columns(4)
        
        with feat_cols[0]:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🎯</div>
                <div class='feature-title'>High Accuracy</div>
                <div class='feature-desc'>65%+ accuracy on test dataset with optimized CNN architecture</div>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_cols[1]:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>⚡</div>
                <div class='feature-title'>Fast Prediction</div>
                <div class='feature-desc'>Get results in seconds with our optimized model</div>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_cols[2]:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>📊</div>
                <div class='feature-title'>Detailed Insights</div>
                <div class='feature-desc'>View waveforms, MFCC features, and top 5 predictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_cols[3]:
            st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>💾</div>
                <div class='feature-title'>Export Results</div>
                <div class='feature-desc'>Download predictions as CSV for further analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Stats Display
        st.markdown("<br><br><h2 style='text-align: center; color: #667eea;'>📈 Model Statistics</h2>", unsafe_allow_html=True)
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.markdown("""
            <div class='feature-card'>
                <div class='stat-number'>114</div>
                <div class='stat-label'>Bird Species</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[1]:
            st.markdown("""
            <div class='feature-card'>
                <div class='stat-number'>40</div>
                <div class='stat-label'>MFCC Features</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[2]:
            st.markdown("""
            <div class='feature-card'>
                <div class='stat-number'>65%</div>
                <div class='stat-label'>Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[3]:
            st.markdown("""
            <div class='feature-card'>
                <div class='stat-number'>CNN</div>
                <div class='stat-label'>Architecture</div>
            </div>
            """, unsafe_allow_html=True)
        
        # How It Works Section
        st.markdown("<br><br><h2 style='text-align: center; color: #667eea;'>🔬 How It Works</h2>", unsafe_allow_html=True)
        
        process_cols = st.columns(5)
        
        with process_cols[0]:
            st.markdown("""
            <div class='step-card'>
                <div style='font-size: 2.5rem;'>📤</div>
                <h4 style='color: #5a3fa0; margin-top: 0.8rem; margin-bottom: 0.4rem;'>1. Upload</h4>
                <p style='color: #555; font-size: 0.9rem; margin: 0;'>Upload your bird sound file</p>
            </div>
            """, unsafe_allow_html=True)

        with process_cols[1]:
            st.markdown("""
            <div class='step-card'>
                <div style='font-size: 2.5rem;'>🎼</div>
                <h4 style='color: #5a3fa0; margin-top: 0.8rem; margin-bottom: 0.4rem;'>2. Extract</h4>
                <p style='color: #555; font-size: 0.9rem; margin: 0;'>Extract MFCC features</p>
            </div>
            """, unsafe_allow_html=True)

        with process_cols[2]:
            st.markdown("""
            <div class='step-card'>
                <div style='font-size: 2.5rem;'>🧠</div>
                <h4 style='color: #5a3fa0; margin-top: 0.8rem; margin-bottom: 0.4rem;'>3. Analyze</h4>
                <p style='color: #555; font-size: 0.9rem; margin: 0;'>CNN processes features</p>
            </div>
            """, unsafe_allow_html=True)

        with process_cols[3]:
            st.markdown("""
            <div class='step-card'>
                <div style='font-size: 2.5rem;'>🎯</div>
                <h4 style='color: #5a3fa0; margin-top: 0.8rem; margin-bottom: 0.4rem;'>4. Predict</h4>
                <p style='color: #555; font-size: 0.9rem; margin: 0;'>Identify bird species</p>
            </div>
            """, unsafe_allow_html=True)

        with process_cols[4]:
            st.markdown("""
            <div class='step-card'>
                <div style='font-size: 2.5rem;'>📊</div>
                <h4 style='color: #5a3fa0; margin-top: 0.8rem; margin-bottom: 0.4rem;'>5. Results</h4>
                <p style='color: #555; font-size: 0.9rem; margin: 0;'>View detailed insights</p>
            </div>
            """, unsafe_allow_html=True)

    # Professional Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='footer'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>🎵 KooKoo AI</h3>
        <p style='color: #666;'>Powered by <strong>TensorFlow 2.x</strong> & <strong>Streamlit</strong></p>
        <p style='color: #999; font-size: 0.9rem; margin-top: 1rem;'>
            © 2025 KooKoo AI | Made with ❤️ for bird enthusiasts and researchers
        </p>
        <div style='margin-top: 1.5rem;'>
            <a href='https://github.com/meenaprajapat/Bird-Sound-Classification' target='_blank' style='color: #667eea; text-decoration: none; margin: 0 1rem; font-weight: 600;'>
                📁 GitHub Repository
            </a>
            <a href='mailto:meenaprajapat98132@gmail.com' style='color: #667eea; text-decoration: none; margin: 0 1rem; font-weight: 600;'>
                📧 Contact
            </a>
            <a href='https://www.linkedin.com/in/meena-a166b4200' target='_blank' style='color: #667eea; text-decoration: none; margin: 0 1rem; font-weight: 600;'>
                🔗 LinkedIn
            </a>
        </div>
        <p style='color: #999; font-size: 0.85rem; margin-top: 1.5rem;'>
            Developed by <strong>Meena Prajapat</strong> | 
            <a href='https://github.com/meenaprajapat' target='_blank' style='color: #667eea; text-decoration: none;'>meenaprajapat</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
