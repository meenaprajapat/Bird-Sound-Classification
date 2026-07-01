"""Injects the custom glassmorphism CSS theme into the Streamlit app."""

import streamlit as st

_CSS = r"""
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
"""


def inject_styles():
    """Apply the app-wide custom CSS. Call once near the top of the app."""
    st.markdown(_CSS, unsafe_allow_html=True)
