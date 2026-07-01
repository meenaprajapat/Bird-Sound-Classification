import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.styles import inject_styles
from src.model_loader import load_model_and_labels
from src.audio_processing import AudioValidationError
from src.predictor import predict_bird_species
from src.visualizations import plot_waveform, plot_mfcc, plot_top_predictions

# Page configuration
st.set_page_config(
    page_title="🐦 KooKoo AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject the custom glassmorphism theme
inject_styles()


@st.cache_resource
def load_model_and_dict():
    """Cached loader used by the UI. Returns (model, labels) or (None, None) on error."""
    try:
        return load_model_and_labels()
    except Exception as e:
        st.error(f"❌ Error loading model or labels: {e}")
        return None, None


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

    # Sample audio — let visitors try the app with one click
    sample_path = Path("sample_bird_call.ogg")
    if uploaded_file is None and sample_path.exists():
        st.markdown("<p style='text-align:center; color:#5a3fa0; margin: 0.5rem 0;'>— or —</p>", unsafe_allow_html=True)
        if st.button("🎵 Try a Sample Bird Call", use_container_width=True):
            st.session_state["use_sample"] = True
        if st.session_state.get("use_sample"):
            st.info("🎧 Using sample recording (**Emu call**). Click **Classify Bird Species** below to see the model in action!")

    # Resolve the active audio source: uploaded file OR the bundled sample
    active_audio = uploaded_file
    active_name = uploaded_file.name if uploaded_file is not None else None
    if uploaded_file is None and st.session_state.get("use_sample") and sample_path.exists():
        active_audio = sample_path
        active_name = sample_path.name

    if active_audio is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        # Read the raw audio bytes once — works for both an upload and the sample file
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getbuffer()
            file_size_kb = uploaded_file.size / 1024
            file_type = uploaded_file.type
        else:
            audio_bytes = active_audio.read_bytes()
            file_size_kb = len(audio_bytes) / 1024
            file_type = "audio/ogg"

        with col1:
            st.markdown("### 🎧 Audio Player")
            st.audio(bytes(audio_bytes))

            # Display file info
            file_details = {
                "Filename": active_name,
                "File size": f"{file_size_kb:.2f} KB",
                "File type": file_type
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
                # Save the audio bytes to a temp path for librosa to read
                suffix = Path(active_name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    temp_audio_path = tmp.name

                try:
                    try:
                        result = predict_bird_species(temp_audio_path, model, prediction_dict)
                        predicted_species = result["species"]
                        confidence = result["confidence"]
                        top_5_predictions = result["top_k"]
                        audio = result["audio"]
                        sr = result["sample_rate"]
                        mfccs_full = result["mfccs"]
                    except AudioValidationError as ve:
                        st.warning(f"⚠️ {ve}")
                        predicted_species = None
                    except Exception as ex:
                        st.error(
                            "❌ Could not process this audio file. It may be corrupted "
                            "or in an unsupported format.\n\n"
                            f"Details: {ex}"
                        )
                        predicted_species = None
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

                    # Confidence-based guidance
                    if confidence < 20:
                        st.warning(
                            "⚠️ **Low confidence.** This may not be a clear bird call, or the species "
                            "might be outside the 114 the model was trained on. Try a cleaner recording "
                            "with a single, prominent bird call."
                        )
                    elif confidence < 45:
                        st.info(
                            "ℹ️ **Moderate confidence.** Check the Top-5 predictions below — the correct "
                            "species may be among them."
                        )

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
                            fig_top5 = plot_top_predictions(top_5_predictions)
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
                        fig_mfcc = plot_mfcc(mfccs_full)
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
                        file_name=f"predictions_{active_name}.csv",
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
