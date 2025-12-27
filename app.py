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
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Upload Section */
    .upload-section {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 3px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.2);
    }
    
    .upload-text {
        text-align: center;
        padding: 2rem;
    }
    
    .upload-text h2 {
        color: #667eea;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 50px rgba(245, 87, 108, 0.3);
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
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card h4 {
        color: #667eea;
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
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #00bcd4;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .info-box h3 {
        color: #00838f;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        margin-top: 3rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    .footer p {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Audio Player */
    .stAudio {
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #666;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and prediction dictionary
@st.cache_resource
def load_model_and_dict():
    """Load the trained model and prediction dictionary"""
    try:
        model_path = Path("model (2).h5")
        json_path = Path("prediction.json")
        
        if not model_path.exists():
            st.error("❌ model (2).h5 not found in the current directory!")
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
        <p>Advanced Deep Learning System for Identifying 114 Bird Species</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with Professional Design
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Logo/Icon
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 5rem;'>🐦</div>
                <h2 style='color: white; margin-top: 0.5rem;'>KooKoo AI</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
        
        # About Section
        st.markdown("""
        <div style='color: white; padding: 1rem;'>
            <h3 style='color: white;'>📚 About the Model</h3>
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <p><strong>🧠 Architecture:</strong> Optimized CNN</p>
                <p><strong>🎼 Features:</strong> 40 MFCC Coefficients</p>
                <p><strong>🐦 Species:</strong> 114 Bird Types</p>
                <p><strong>⚡ Framework:</strong> TensorFlow 2.x</p>
                <p><strong>📊 Accuracy:</strong> ~65%+ on Test Set</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
        
        # How to Use
        st.markdown("""
        <div style='color: white; padding: 1rem;'>
            <h3 style='color: white;'>🎯 Quick Guide</h3>
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <ol style='padding-left: 1.2rem;'>
                    <li>📤 Upload bird sound file</li>
                    <li>🎧 Listen to preview</li>
                    <li>🚀 Click "Classify Species"</li>
                    <li>📊 View detailed results</li>
                    <li>💾 Download predictions</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
        
        # Supported Formats
        st.markdown("""
        <div style='color: white; padding: 1rem;'>
            <h3 style='color: white;'>📁 File Formats</h3>
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <p>✅ MP3 - Most common</p>
                <p>✅ WAV - Uncompressed</p>
                <p>✅ OGG - Open format</p>
                <p>✅ FLAC - Lossless</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Footer in Sidebar
        st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.7); font-size: 0.85rem; padding: 1rem;'>
            <p>Made with ❤️ using</p>
            <p><strong>TensorFlow & Streamlit</strong></p>
            <p style='margin-top: 1rem;'>© 2025 KooKoo AI</p>
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
                # Save uploaded file temporarily
                temp_audio_path = f"temp_{uploaded_file.name}"
                with open(temp_audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Make prediction
                predicted_species, confidence, top_5_predictions, audio, sr, mfccs_full = predict_bird_species(
                    temp_audio_path, model, prediction_dict
                )
                
                # Remove temporary file
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
            <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08);'>
                <div style='font-size: 2.5rem; color: #667eea;'>📤</div>
                <h4 style='color: #667eea; margin-top: 1rem;'>1. Upload</h4>
                <p style='color: #666; font-size: 0.9rem;'>Upload your bird sound file</p>
            </div>
            """, unsafe_allow_html=True)
        
        with process_cols[1]:
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08);'>
                <div style='font-size: 2.5rem; color: #764ba2;'>🎼</div>
                <h4 style='color: #764ba2; margin-top: 1rem;'>2. Extract</h4>
                <p style='color: #666; font-size: 0.9rem;'>Extract MFCC features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with process_cols[2]:
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08);'>
                <div style='font-size: 2.5rem; color: #667eea;'>🧠</div>
                <h4 style='color: #667eea; margin-top: 1rem;'>3. Analyze</h4>
                <p style='color: #666; font-size: 0.9rem;'>CNN processes features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with process_cols[3]:
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08);'>
                <div style='font-size: 2.5rem; color: #764ba2;'>🎯</div>
                <h4 style='color: #764ba2; margin-top: 1rem;'>4. Predict</h4>
                <p style='color: #666; font-size: 0.9rem;'>Identify bird species</p>
            </div>
            """, unsafe_allow_html=True)
        
        with process_cols[4]:
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08);'>
                <div style='font-size: 2.5rem; color: #667eea;'>📊</div>
                <h4 style='color: #667eea; margin-top: 1rem;'>5. Results</h4>
                <p style='color: #666; font-size: 0.9rem;'>View detailed insights</p>
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
            <a href='https://github.com/tarun-02005/koo-koo-ai-new' target='_blank' style='color: #667eea; text-decoration: none; margin: 0 1rem; font-weight: 600;'>
                📁 GitHub Repository
            </a>
            <a href='mailto:meenaprajapat98132@gmail.com' style='color: #667eea; text-decoration: none; margin: 0 1rem; font-weight: 600;'>
                📧 Contact
            </a>
            <a href='https://www.linkedin.com/in/meena-prajapat-a166b4200/' target='_blank' style='color: #667eea; text-decoration: none; margin: 0 1rem; font-weight: 600;'>
                � LinkedIn
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
