"""TomatoCare AI: a Streamlit prototype for tomato leaf classification."""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "tomato_disease_cnn.h5"
if not MODEL_PATH.is_file():
    MODEL_PATH = Path(__file__).resolve().parent / "model" / "tomato_disease_cnn.h5"

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# severity drives the color-coded status badge shown with a prediction:
#   "none"     -> healthy / no concern
#   "moderate" -> manageable with routine care
#   "high"     -> fast-moving / high-impact, worth prompt attention
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "severity": "moderate",
        "description": "A bacterial disease that can affect tomato foliage and fruit.",
        "symptoms": "Small, dark, water-soaked spots that may develop yellow halos.",
        "management": "Use clean seed and tools, avoid working with wet plants, and remove affected plant material where appropriate.",
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "severity": "moderate",
        "description": "A common fungal disease that often appears first on older leaves.",
        "symptoms": "Dark spots with concentric rings; surrounding leaf tissue may yellow.",
        "management": "Remove affected leaves, improve airflow, and avoid overhead watering where practical.",
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "severity": "high",
        "description": "A fast-spreading disease that can affect leaves, stems, and fruit in favorable conditions.",
        "symptoms": "Irregular, dark lesions; pale growth may appear on leaf undersides in humid weather.",
        "management": "Monitor plants closely, limit leaf wetness, and remove suspect plant material promptly.",
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "severity": "moderate",
        "description": "A fungal disease favored by humid conditions and limited air circulation.",
        "symptoms": "Pale green or yellow patches on upper leaf surfaces, sometimes with olive growth underneath.",
        "management": "Increase ventilation, reduce humidity around foliage, and remove heavily affected leaves.",
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "severity": "moderate",
        "description": "A fungal leaf disease that commonly begins on lower, older foliage.",
        "symptoms": "Many small circular spots with pale centers and dark borders; leaves may yellow and drop.",
        "management": "Remove affected lower leaves, keep foliage dry, and clear plant debris after the season.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mites",
        "severity": "moderate",
        "description": "Tiny plant-feeding mites that can cause visible leaf damage, especially in hot, dry conditions.",
        "symptoms": "Fine pale stippling, bronzing, and sometimes delicate webbing on leaf undersides.",
        "management": "Inspect leaf undersides, reduce dusty or water-stressed conditions, and use locally recommended controls if needed.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "severity": "high",
        "description": "A viral disease that may affect leaf color, shape, and plant growth.",
        "symptoms": "Mottled light and dark green areas, leaf distortion, or reduced growth.",
        "management": "Sanitize hands and tools, remove suspect plants carefully, and use clean planting material.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "severity": "high",
        "description": "A viral disease commonly associated with whitefly transmission.",
        "symptoms": "Upward-curling yellow leaves, smaller new growth, and reduced fruit set.",
        "management": "Monitor for whiteflies, remove suspect plants when appropriate, and follow local integrated pest-management guidance.",
    },
    "Tomato___healthy": {
        "name": "Healthy",
        "severity": "none",
        "description": "The image is most consistent with the healthy-leaf class in this model.",
        "symptoms": "No target disease pattern was strongly identified in the submitted image.",
        "management": "Continue routine monitoring and good growing practices; a healthy prediction does not rule out other issues.",
    },
}

SEVERITY_STYLE = {
    "none": {"label": "Healthy", "color": "#34d399", "bg": "rgba(52, 211, 153, 0.12)"},
    "moderate": {"label": "Moderate risk", "color": "#fbbf24", "bg": "rgba(251, 191, 36, 0.12)"},
    "high": {"label": "High risk", "color": "#f87171", "bg": "rgba(248, 113, 113, 0.12)"},
}

NAV_ITEMS = ["Home", "Detection", "Model Info", "About"]
NAV_ICONS = {"Home": "🏠", "Detection": "🔍", "Model Info": "📊", "About": "ℹ️"}


st.set_page_config(
    page_title="TomatoCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

    :root {
        --bg-0: #0b0f0d;
        --bg-1: #0f1512;
        --surface: #131a16;
        --surface-2: #171f1a;
        --line: #23302a;
        --line-soft: #1b2420;
        --ink: #eef3ef;
        --muted: #9fb0a6;
        --muted-2: #6f8478;
        --primary: #34d399;
        --primary-dark: #10b981;
        --primary-deep: #0e6b48;
        --primary-deep-hover: #0b5a3c;
        --primary-soft: rgba(52, 211, 153, 0.12);
        --accent: #7dd3c0;
        --radius-lg: 18px;
        --radius-md: 14px;
        --radius-sm: 10px;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        --shadow-soft: 0 6px 18px rgba(0, 0, 0, 0.18);
    }

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 12% -10%, rgba(52, 211, 153, 0.09) 0%, transparent 45%),
            radial-gradient(circle at 100% 0%, rgba(125, 211, 192, 0.05) 0%, transparent 40%),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 15, 13, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }
    .block-container { max-width: 1180px; padding-top: 70px !important; padding-bottom: 5rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0d1310;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
        display: flex; flex-direction: column; min-height: 100vh;
    }
    .brand { display: flex; align-items: center; gap: .6rem; padding-bottom: 1rem; border-bottom: 1px solid var(--line-soft); margin-bottom: 1.1rem; }
    .brand-mark {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
    }
    .brand-text { display: flex; flex-direction: column; line-height: 1.15; }
    .brand-title { font: 700 1.02rem 'Poppins', sans-serif; color: var(--ink); white-space: nowrap; }
    .brand-sub { font-size: .72rem; color: var(--muted-2); letter-spacing: .04em; text-transform: uppercase; white-space: nowrap; }
    .sidebar-status {
        margin-top: 1.1rem; padding: .8rem .9rem;
        background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm);
    }
    .sidebar-status-row { display: flex; align-items: center; gap: .5rem; font-size: .82rem; color: var(--muted); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--primary); box-shadow: 0 0 0 3px var(--primary-soft); flex-shrink: 0; }
    .sidebar-footer {
        margin-top: auto; padding: 1.2rem 0 1.5rem; border-top: 1px solid var(--line-soft);
        font-size: .82rem; color: var(--muted-2); line-height: 1.5;
    }
    .sidebar-footer strong { color: var(--muted); }
    .sidebar-image { border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--line);
        margin: 1rem 0; box-shadow: var(--shadow-soft); }

    /* ---------- Top navigation ---------- */
    .st-key-topnav {
        padding: .75rem 1rem; background: var(--surface); border: 1px solid var(--line);
        border-radius: var(--radius-md); box-shadow: var(--shadow-soft);
        margin: 0 auto 1.4rem; max-width: 800px;
    }
    .st-key-topnav [data-testid="stHorizontalBlock"] { gap: .35rem; align-items: center !important; }
    .st-key-topnav [data-testid="stColumn"] { display: flex !important; align-items: center !important; justify-content: center !important; min-height: 44px; }
    .st-key-topnav [data-testid="stMarkdownContainer"] { margin: 0 !important; padding: 0 !important; width: 100%; }
    .st-key-topnav [data-testid="stMarkdownContainer"] > div { margin: 0 !important; }
    .st-key-topnav div.stButton { width: 100%; display: flex; justify-content: center; align-items: center; margin: 0 !important; }
    .st-key-topnav div.stButton > button {
        font-weight: 600; font-size: .85rem !important;
        padding: .48rem .8rem !important; margin: 0 !important;
        border-radius: 999px !important; min-height: 38px !important;
        white-space: nowrap !important; width: 100%;
        transition: all .15s ease;
    }
    .st-key-topnav div.stButton > button[kind="secondary"] {
        background: transparent !important; color: var(--muted) !important;
        border: 1px solid transparent !important; box-shadow: none !important;
    }
    .st-key-topnav div.stButton > button[kind="secondary"]:hover {
        color: var(--ink) !important; background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important; transform: none;
    }
    .st-key-topnav div.stButton > button[kind="primary"] {
        color: var(--primary) !important; background: var(--primary-soft) !important;
        border: 1px solid rgba(52, 211, 153, 0.3) !important; box-shadow: none !important;
    }
    .st-key-topnav div.stButton > button[kind="primary"]:hover {
        color: var(--primary) !important; background: rgba(52, 211, 153, 0.2) !important;
        transform: none; box-shadow: none !important;
    }
    .topnav-brand {
        display: flex; align-items: center; justify-content: flex-start; gap: .45rem;
        font: 700 1.05rem 'Poppins', sans-serif; color: var(--ink);
        white-space: nowrap !important; overflow: visible;
        width: 100%; margin: 0 !important; padding: 0 !important;
        line-height: 1 !important; height: 100%;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 { font-family: 'Poppins', 'Inter', sans-serif; color: var(--ink); letter-spacing: -0.01em; }
    h1 { font-size: 2.05rem; font-weight: 700; }
    h2 { font-size: 1.32rem; font-weight: 600; }
    p, li, span { color: var(--muted); }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        padding: clamp(1.8rem, 3.6vw, 3rem);
        border-radius: var(--radius-lg);
        background: linear-gradient(120deg, #132a20 0%, #101a15 55%, #142622 100%);
        border: 1px solid var(--line);
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }
    .hero::after {
        content: ""; position: absolute; inset: 0;
        background: radial-gradient(circle at 88% 10%, rgba(52,211,153,0.14) 0%, transparent 45%);
        pointer-events: none;
    }
    .hero h1 { font-size: clamp(1.8rem, 3vw, 2.5rem); margin: 0 0 .55rem; max-width: 600px; }
    .hero-desc { max-width: 560px; font-size: .98rem; line-height: 1.6; margin: 0 0 1.2rem; position: relative; }
    .hero-stats { display: flex; gap: 1.8rem; margin-top: 1.4rem; margin-bottom: 1.6rem !important; flex-wrap: wrap; }
    .hero-stat-num { font: 700 1.3rem 'Poppins', sans-serif; color: var(--ink); }
    .hero-stat-label { font-size: .77rem; color: var(--muted-2); }

    .st-key-hero_container div.stButton { margin-top: .4rem; }
    .st-key-hero_container div.stButton > button,
    div.stButton > button[key="hero_cta"] {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: .95rem !important;
        padding: .65rem 1.65rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(52, 211, 153, 0.45) !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.38) !important;
        transition: all .2s ease-in-out !important;
    }
    .st-key-hero_container div.stButton > button:hover {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
        box-shadow: 0 6px 22px rgba(52, 211, 153, 0.55) !important;
        transform: translateY(-2px) !important;
    }

    /* ---------- Section labels & dividers ---------- */
    .section-label {
        display: flex; align-items: center; gap: .6rem;
        color: var(--ink); font: 600 1.1rem 'Poppins', sans-serif;
        margin: 1.6rem 0 .8rem;
    }
    .section-label::before {
        content: ""; width: 4px; height: 1.1rem; border-radius: 2px;
        background: linear-gradient(180deg, var(--primary), var(--primary-dark));
        display: inline-block; flex-shrink: 0;
    }
    .section-sub { color: var(--muted-2); font-size: .88rem; margin: 0 0 1.1rem 1rem; }
    .leaf-divider {
        display: flex; align-items: center; gap: .7rem; margin: 2.2rem 0 1.5rem; color: var(--muted-2);
        font-weight: 500; font-size: .92rem;
    }
    .leaf-divider::before, .leaf-divider::after { content: ""; flex: 1; height: 1px; background: var(--line); }

    /* ---------- Feature cards ---------- */
    [data-testid="stHorizontalBlock"]:has(.feature-strip) {
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:has(.feature-strip) {
        display: flex; flex-direction: column; flex: 1 1 0px;
    }
    .feature-strip {
        background: var(--surface);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: var(--radius-md);
        padding: 1.35rem 1.25rem;
        height: 100%;
        min-height: 165px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-shadow: var(--shadow-soft);
        transition: border-color .2s ease, background .2s ease, transform .2s ease, box-shadow .2s ease;
    }
    .feature-strip:hover {
        background: var(--surface-2);
        border-color: rgba(52, 211, 153, 0.75);
        box-shadow: 0 8px 20px rgba(52, 211, 153, 0.12);
        transform: translateY(-2px);
    }
    .feature-strip h3 { font-size: .98rem; margin: 0 0 .45rem; color: var(--ink); display: flex; align-items: center; gap: .45rem; font-weight: 600; }
    .feature-strip p { margin: 0; font-size: .87rem; line-height: 1.55; color: var(--muted); }
    .feature-icon-inline { font-size: 1.1rem; flex-shrink: 0; }

    /* ---------- How it works ---------- */
    .timeline { display: flex; flex-direction: column; gap: 0; margin-top: .5rem; }
    .timeline-step { display: flex; gap: 1rem; padding-bottom: 1.5rem; position: relative; }
    .timeline-step:last-child { padding-bottom: 0; }
    .timeline-step::before {
        content: ""; position: absolute; left: 17px; top: 38px; bottom: 0; width: 2px;
        background: var(--line);
    }
    .timeline-step:last-child::before { display: none; }
    .timeline-num {
        flex-shrink: 0; width: 36px; height: 36px; border-radius: 50%;
        background: var(--primary-soft); border: 1px solid rgba(52,211,153,0.32);
        color: var(--primary); font: 700 .95rem 'Poppins', sans-serif;
        display: flex; align-items: center; justify-content: center; z-index: 1;
    }
    .timeline-body h3 { font-size: .98rem; margin: .1rem 0 .25rem; color: var(--ink); }
    .timeline-body p { margin: 0; font-size: .87rem; line-height: 1.5; }

    /* ---------- Plain info sections ---------- */
    .plain-section { border-left: 2px solid var(--line); padding-left: 1.05rem; margin-bottom: 1rem; }
    .plain-section h3 { font-size: 1rem; margin: 0 0 .3rem; color: var(--ink); }
    .plain-section p { margin: 0; font-size: .89rem; line-height: 1.55; }

    .info-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-md);
        box-shadow: var(--shadow-soft); padding: 1.3rem 1.4rem;
    }
    .info-card h3 { font-size: 1.02rem; margin: 0 0 .4rem; color: var(--ink); }
    .info-card p { margin: 0; font-size: .89rem; line-height: 1.55; }

    hr.divider { border: none; border-top: 1px solid var(--line); margin: .85rem 0; }

    /* ---------- Result card ---------- */
    .result-card {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
        border: 1px solid var(--line); border-radius: var(--radius-lg);
        padding: 1.6rem 1.8rem; box-shadow: var(--shadow);
    }
    .result-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: .8rem; }
    .result-label { font-size: .78rem; color: var(--muted-2); font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    .result-name { color: var(--ink); font: 700 1.65rem 'Poppins', sans-serif; margin: .3rem 0 .15rem; }
    .status-pill {
        display: inline-flex; align-items: center; gap: .4rem;
        padding: .32rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600; white-space: nowrap;
    }
    .status-pill .dot { box-shadow: none; }
    .confidence-row { display: flex; align-items: baseline; gap: .45rem; margin: 1.05rem 0 .35rem; }
    .confidence-text { font-size: 1.05rem; font-weight: 600; color: var(--ink); }
    .confidence-value { font: 700 1.25rem 'Poppins', sans-serif; color: var(--primary); }
    .disclaimer {
        font-size: .82rem; color: var(--muted-2);
        margin: 2rem 0 1rem; padding: .75rem 1rem;
        background: var(--surface); border: 1px solid var(--line-soft);
        border-radius: var(--radius-sm); display: flex; align-items: center; gap: .5rem;
    }

    /* ---------- Upload dropzone ---------- */
    .st-key-dropzone { border: 1px dashed var(--line); border-radius: var(--radius-md);
        padding: 1.1rem 1.2rem 0.3rem; background: var(--surface); margin-bottom: 1.2rem; }

    [data-testid="stImage"] img { border-radius: var(--radius-md); border: 1px solid var(--line); transition: transform .18s ease; }
    [data-testid="stImage"] img:hover { transform: scale(1.015); }

    /* ---------- Buttons ---------- */
    div.stButton > button[kind="primary"] {
        background: var(--primary-deep);
        border: 1px solid rgba(255,255,255,0.06); color: #ffffff; font-weight: 700;
        border-radius: var(--radius-sm); padding: .65rem 1.3rem;
        box-shadow: var(--shadow-soft);
        transition: background .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: var(--primary-deep-hover); transform: translateY(-1px); box-shadow: var(--shadow);
    }

    /* ---------- Streamlit widget skinning ---------- */
    [data-testid="stMetric"] {
        background: var(--surface); padding: 1.1rem 1.2rem; border: 1px solid var(--line);
        border-radius: var(--radius-md); box-shadow: var(--shadow-soft);
        min-width: 0; overflow: visible;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted-2) !important; font-size: .78rem !important;
        text-transform: uppercase; letter-spacing: .04em;
        white-space: normal !important; word-break: break-word !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink) !important; font-size: 1.22rem !important;
        font-weight: 700 !important; white-space: normal !important;
        word-break: break-word !important; overflow-wrap: anywhere !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.22rem !important; white-space: normal !important;
        overflow: visible !important; text-overflow: clip !important;
    }
    [data-testid="stFileUploader"] section { background: var(--surface-2); border-color: var(--line); border-radius: var(--radius-sm); }
    [data-testid="stFileUploader"] section:hover { border-color: var(--primary); }
    [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, var(--primary), var(--primary-dark)); }
    [data-testid="stCaptionContainer"] { color: var(--muted-2) !important; }
    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); }

    @media (max-width: 640px) {
        .block-container { padding: .8rem .8rem 2rem; }
        h1 { font-size: 1.55rem; }
        .hero { padding: 1.3rem; border-radius: 14px; }
        .hero-stats { gap: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# ML helpers (unchanged logic)
# ----------------------------------------------------------------------------

@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the trained model once and reuse it across Streamlit reruns."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        # Fallback for Keras 3 HDF5 weights-only files without saved model config metadata
        import h5py
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(9, activation="softmax")
        ])
        dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
        model(dummy, training=False)
        
        with h5py.File(MODEL_PATH, "r") as f:
            if "layers/conv2d/vars/0" in f:
                model.get_layer(index=0).set_weights([np.array(f["layers/conv2d/vars/0"]), np.array(f["layers/conv2d/vars/1"])])
                model.get_layer(index=2).set_weights([np.array(f["layers/conv2d_1/vars/0"]), np.array(f["layers/conv2d_1/vars/1"])])
                model.get_layer(index=4).set_weights([np.array(f["layers/conv2d_2/vars/0"]), np.array(f["layers/conv2d_2/vars/1"])])
                model.get_layer(index=7).set_weights([np.array(f["layers/dense/vars/0"]), np.array(f["layers/dense/vars/1"])])
                model.get_layer(index=8).set_weights([np.array(f["layers/dense_1/vars/0"]), np.array(f["layers/dense_1/vars/1"])])
            else:
                model.load_weights(MODEL_PATH)
        return model


def display_name(class_name: str) -> str:
    """Return a readable name for a model class label."""
    return DISEASE_INFO[class_name]["name"]


def decode_image(image_bytes: bytes) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """Validate an uploaded image and prepare CNN model input."""
    image = Image.open(io.BytesIO(image_bytes))
    original = image.convert("RGB")
    resized = original.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    batch = np.asarray(resized, dtype=np.float32) / 255.0
    model_input = np.expand_dims(batch, axis=0)
    return original, resized, model_input


def predict(model: tf.keras.Model, model_input: np.ndarray) -> np.ndarray:
    """Run inference and return normalized probabilities for all classes."""
    output = np.asarray(model.predict(model_input, verbose=0), dtype=np.float64)
    if output.ndim == 2 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 1 or output.size != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} class scores; received shape {output.shape}.")
    if not np.all(np.isfinite(output)):
        raise ValueError("The model returned a non-finite prediction.")

    if np.all(output >= 0) and np.isclose(output.sum(), 1.0, atol=1e-3):
        probabilities = output / output.sum()
    else:
        shifted = output - np.max(output)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum()
    return probabilities


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------

@st.cache_data
def list_local_images() -> list[str]:
    """Return sorted paths of images already present in the project's images/ folder."""
    if not IMAGES_DIR.is_dir():
        return []
    return sorted(
        str(p) for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_named_image(name: str, images: list[str]) -> str | None:
    """Find an image by exact filename first, falling back to a case-insensitive stem match."""
    exact = IMAGES_DIR / name
    if exact.is_file():
        return str(exact)
    target_stem = Path(name).stem.lower()
    for image_path in images:
        if Path(image_path).stem.lower() == target_stem:
            return image_path
    return None


def set_page(page: str) -> None:
    st.session_state["navigation"] = page


def status_pill_html(severity: str) -> str:
    style = SEVERITY_STYLE[severity]
    return (
        f'<span class="status-pill" style="background:{style["bg"]}; color:{style["color"]};'
        f' border:1px solid {style["color"]}44;">'
        f'<span class="dot" style="background:{style["color"]};"></span>{style["label"]}</span>'
    )


def render_topnav() -> None:
    current = st.session_state.get("navigation", "Home")
    with st.container(key="topnav"):
        columns = st.columns([1.35, 1.0, 1.1, 1.2, 1.0])
        with columns[0]:
            st.markdown('<div class="topnav-brand">🍅 TomatoCare AI</div>', unsafe_allow_html=True)
        for column, item in zip(columns[1:], NAV_ITEMS):
            with column:
                st.button(
                    f"{NAV_ICONS[item]} {item}",
                    key=f"nav_{item}",
                    type="primary" if current == item else "secondary",
                    use_container_width=True,
                    on_click=set_page,
                    args=(item,),
                )


if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"

local_images = list_local_images()
tomato_2_image = find_named_image("tomato2.jpeg", local_images)
tomato_named_images = [p for p in local_images if "tomato" in Path(p).stem.lower()]
if not tomato_named_images:
    tomato_named_images = local_images[:2]
home_gallery_images = tomato_named_images[:2]

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🍅</div>
            <div class="brand-text">
                <div class="brand-title">TomatoCare AI</div>
                <div class="brand-sub">Leaf Diagnostics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if tomato_2_image:
        st.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
        st.image(tomato_2_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-status">
            <div class="sidebar-status-row"><span class="dot"></span> Model ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Computer-vision prototype for tomato leaf images")
    st.markdown(
        '<div class="sidebar-footer">💙 <strong>Made by Group 9</strong></div>',
        unsafe_allow_html=True,
    )

render_topnav()
page = st.session_state["navigation"]


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

if page == "Home":
    with st.container(key="hero_container"):
        st.markdown(
            """
            <h1>Understand tomato leaf health at a glance</h1>
            <p class="hero-desc">TomatoCare AI is a computer-vision prototype that screens a photo of a
            tomato leaf against nine trained categories, then returns a confidence score together with
            general background and management guidance for the predicted class.</p>
            <div class="hero-stats">
                <div><div class="hero-stat-num">9</div><div class="hero-stat-label">Leaf classes</div></div>
                <div><div class="hero-stat-num">128²</div><div class="hero-stat-label">Input resolution</div></div>
                <div><div class="hero-stat-num">Custom CNN</div><div class="hero-stat-label">Backbone model</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Try Detection →",
            key="hero_cta",
            type="primary",
            on_click=set_page,
            args=("Detection",),
        )

    st.markdown('<div class="leaf-divider">Tomato Leaves</div>', unsafe_allow_html=True)
    if home_gallery_images:
        gallery_columns = st.columns(len(home_gallery_images))
        for column, image_path in zip(gallery_columns, home_gallery_images):
            with column:
                st.markdown('<div class="hero-img-frame">', unsafe_allow_html=True)
                st.image(image_path, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hero-img-placeholder">🍅</div>', unsafe_allow_html=True)
        st.caption("Add tomato images to the project's images/ folder to feature them here.")

    st.markdown('<div class="leaf-divider">Key Features</div>', unsafe_allow_html=True)
    feature_columns = st.columns(3)
    features = [
        ("🔍", " Pattern-Based Analysis", "A trained image model checks the uploaded leaf against learned visual patterns for each class."),
        ("📊", " Confidence Breakdown", "See the model's confidence for the top prediction alongside the full probability spread."),
        ("📘", " Reference Guidance", "Each predicted class links to a short description, common symptoms, and general management notes."),
    ]
    for column, (icon, title, description) in zip(feature_columns, features):
        with column:
            st.markdown(
                f'<div class="feature-strip"><h3><span class="feature-icon-inline">{icon}</span>{title}</h3>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="leaf-divider">How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("Upload a leaf photo", "Upload a clear image of a tomato plant leaf from the Detection page."),
        ("The model analyzes it", "The AI analyzes the image for learned disease patterns using the trained classifier."),
        ("Review the prediction", "Receive the predicted disease (or healthy result) along with a confidence score."),
        ("Read the guidance", "View general information and recommended management guidance for the predicted condition."),
    ]
    timeline_html = '<div class="timeline">'
    for index, (title, description) in enumerate(steps, start=1):
        timeline_html += (
            f'<div class="timeline-step"><div class="timeline-num">{index}</div>'
            f'<div class="timeline-body"><h3>{title}</h3><p>{description}</p></div></div>'
        )
    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)



elif page == "Detection":
    st.title("Disease Detection")
    st.write("Upload a clear photo of a tomato leaf to get a model prediction with a confidence score.")

    with st.container(key="dropzone"):
        uploaded_file = st.file_uploader(
            "Upload Tomato Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        if Path(uploaded_file.name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            st.error("Unsupported file format. Please upload a JPG, JPEG, or PNG image.")
        else:
            try:
                original_image, processed_image, model_input = decode_image(image_bytes)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                st.error("This file could not be read as an image. Please choose a valid JPG, JPEG, or PNG file.")
            else:
                st.markdown('<div class="section-label">Image Analysis</div>', unsafe_allow_html=True)
                image_columns = st.columns(2)
                with image_columns[0]:
                    st.image(original_image, caption="Original Image", use_container_width=True)
                with image_columns[1]:
                    st.image(processed_image, caption="Processed Image · 128 × 128", use_container_width=True)

                image_hash = hashlib.sha256(image_bytes).hexdigest()
                result_key = f"prediction_{image_hash}"
                if result_key not in st.session_state:
                    if not MODEL_PATH.is_file():
                        st.error(
                            f"The model file `{MODEL_PATH.name}` is missing. Place it beside `app.py` and upload the image again."
                        )
                    else:
                        try:
                            with st.spinner("Analyzing leaf pattern against trained classes..."):
                                model = load_model()
                                probabilities = predict(model, model_input)
                                time.sleep(0.25)  # brief pause so the analyzing state is perceptible
                            st.session_state[result_key] = probabilities.tolist()
                        except FileNotFoundError:
                            st.error(f"The model file `{MODEL_PATH.name}` could not be found.")
                        except Exception as error:
                            st.error(f"Prediction could not be completed: {error}")

                if result_key in st.session_state:
                    probabilities = np.asarray(st.session_state[result_key], dtype=np.float64)
                    best_index = int(np.argmax(probabilities))
                    predicted_class = CLASS_NAMES[best_index]
                    confidence = float(probabilities[best_index])
                    disease = DISEASE_INFO[predicted_class]

                    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-top">
                                <div>
                                    <div class="result-label">Predicted class</div>
                                    <div class="result-name">{disease["name"]}</div>
                                </div>
                                {status_pill_html(disease["severity"])}
                            </div>
                            <div class="confidence-row">
                                <span class="confidence-text">Model confidence - </span>
                                <span class="confidence-value">{confidence:.1%}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(confidence, text=f"Confidence - {confidence:.1%}")

                    st.markdown('<div class="section-label">About this class</div>', unsafe_allow_html=True)
                    info_columns = st.columns(2)
                    with info_columns[0]:
                        st.markdown(
                            f'<div class="plain-section"><h3>Description</h3><p>{disease["description"]}</p></div>'
                            f'<div class="plain-section"><h3>Common symptoms</h3><p>{disease["symptoms"]}</p></div>',
                            unsafe_allow_html=True,
                        )
                    with info_columns[1]:
                        st.markdown(
                            f'<div class="plain-section"><h3>General management</h3><p>{disease["management"]}</p></div>',
                            unsafe_allow_html=True,
                        )

                    sorted_indices = np.argsort(probabilities)[::-1]
                    chart_data = {
                        "Class": [display_name(CLASS_NAMES[index]) for index in sorted_indices],
                        "Probability": [float(probabilities[index]) for index in sorted_indices],
                    }
                    figure = px.bar(
                        chart_data,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        text=px.Constant(""),
                        color_discrete_sequence=["#34d399"],
                    )
                    figure.update_traces(
                        texttemplate="%{x:.1%}",
                        textposition="outside",
                        cliponaxis=False,
                        marker_line_width=0,
                    )
                    figure.update_layout(
                        height=390,
                        margin=dict(l=8, r=55, t=10, b=10),
                        xaxis=dict(
                            title="Probability",
                            tickformat=".0%",
                            range=[0, min(1.12, max(chart_data["Probability"]) * 1.18)],
                            gridcolor="#23302a",
                        ),
                        yaxis=dict(title=None, autorange="reversed"),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d6cd", family="Inter, sans-serif"),
                        bargap=0.35,
                    )
                    st.markdown('<div class="section-label">Prediction Probabilities</div>', unsafe_allow_html=True)
                    st.plotly_chart(figure, use_container_width=True)


elif page == "Model Info":
    st.title("Model Information")
    st.write("Technical details for the classifier used in this prototype.")
    columns = st.columns(4)
    metrics = [
        ("Model Architecture", "Custom CNN", "3-layer Sequential ConvNet"),
        ("Input Size", "128 × 128 × 3", "RGB image tensor"),
        ("Number of Classes", "9", "Tomato leaf categories"),
        ("Framework", "TensorFlow / Keras", "HDF5 saved model (.h5)"),
    ]
    for column, (label, value, help_text) in zip(columns, metrics):
        with column:
            st.metric(label, value, help=help_text)

    st.markdown('<div class="section-label">How the model works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-card"><p>This custom Convolutional Neural Network (CNN) features 3 convolutional '
        'layers with ReLU activation and MaxPooling, followed by Dense classification layers. Input images '
        'are resized to 128 × 128 pixels and normalized by scaling RGB values to [0, 1]. The model was '
        'trained using the Adam optimizer with Sparse Categorical Crossentropy loss across 20 epochs.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Evaluation results</div>', unsafe_allow_html=True)
    score_columns = st.columns(4)
    eval_scores = [("Model Accuracy", "91%"), ("Precision", "92%"), ("Recall", "91%"), ("F1-Score", "92%")]
    for column, (metric, score) in zip(score_columns, eval_scores):
        with column:
            st.metric(metric, score)
    st.caption("Evaluation metrics evaluated on 2,112 held-out test set leaf images.")


else:  # About
    st.title("About TomatoCare AI")
    st.markdown('<div class="section-label">What is TomatoCare AI?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-card"><p>TomatoCare AI is a computer-vision based prototype that classifies tomato '
        'leaf images into predefined categories using a deep-learning model. It is intended for exploration '
        'and educational use, not as a professional agricultural diagnosis.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Technology Stack</div>', unsafe_allow_html=True)
    stack = ["Python", "Streamlit", "TensorFlow / Keras", "Custom CNN", "NumPy", "Pillow", "Plotly"]
    list_items = "".join(f'<li style="margin-bottom: 0.35rem; color: var(--ink);">{item}</li>' for item in stack)
    st.markdown(
        f'<div class="info-card"><ul style="margin: 0; padding-left: 1.2rem; line-height: 1.65;">{list_items}</ul></div>',
        unsafe_allow_html=True,
    )