"""
Streamlit App - Mobile Price Range Prediction
Introduction to Data Science (S2-25_DSECLZG532) - BITS Pilani WILP

Run with:
    streamlit run app.py

This app loads the model artifacts produced by the accompanying Jupyter notebook
(Group2.ipynb) and lets a user enter new mobile phone specifications in the
sidebar to get a live prediction of the price range
(0 = Low, 1 = Medium, 2 = High, 3 = Very High) — all on a single screen, no
scrolling required in the main panel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mobile Price Range Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark gradient theme, compact spacing, card styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 10% 0%, #1b2a4a 0%, #0d1117 45%, #0a0e14 100%);
        color: #e6edf3;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11182b 0%, #0d1117 100%);
        border-right: 1px solid rgba(120,150,255,0.15);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    h1, h2, h3, p, label, span, div { color: #e6edf3; }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.15;
    }
    .hero-sub {
        color: #9aa7b8;
        font-size: 0.92rem;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }

    .result-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .price-value {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.1rem 0 0.2rem 0;
    }

    .price-label {
        font-size: 1.05rem;
        color: #9aa7b8;
        margin-bottom: 0.6rem;
    }

    .model-chip {
        display: inline-block;
        background: rgba(125,211,252,0.12);
        border: 1px solid rgba(125,211,252,0.35);
        color: #7dd3fc;
        border-radius: 8px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #7dd3fc, #a78bfa);
        color: #0d1117;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1rem;
        width: 100%;
        transition: transform 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
        color: #0d1117;
    }

    .stSlider label, .stSelectbox label { font-size: 0.82rem !important; color: #b7c2d0 !important; }

    hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load persisted artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('mobile_price_model.pkl')
    scaler = joblib.load('mobile_price_scaler.pkl')
    feature_cols = joblib.load('mobile_price_features.pkl')
    best_model_name = joblib.load('mobile_price_best_model_name.pkl')
    return model, scaler, feature_cols, best_model_name

model, scaler, feature_cols, best_model_name = load_artifacts()

PRICE_LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
PRICE_COLORS = {0: "#34d399", 1: "#7dd3fc", 2: "#facc15", 3: "#f472b6"}

# ---------------------------------------------------------------------------
# Sidebar — all inputs live here (scrolls independently of the main panel)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Phone Specifications")
    st.caption("Adjust the specs, then hit **Predict** on the right.")

    t1, t2, t3 = st.tabs(["🔋 Core", "📷 Camera/Mem", "📶 Connectivity"])

    with t1:
        battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1200)
        ram = st.slider("RAM (MB)", 250, 4000, 2000)
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
        n_cores = st.slider("Number of Cores", 1, 8, 4)
        talk_time = st.slider("Talk Time (hrs)", 2, 20, 10)

    with t2:
        int_memory = st.slider("Internal Memory (GB)", 2, 64, 32)
        pc = st.slider("Primary Camera (MP)", 0, 20, 10)
        fc = st.slider("Front Camera (MP)", 0, 20, 5)
        px_height = st.slider("Pixel Height", 0, 1960, 800)
        px_width = st.slider("Pixel Width", 500, 2000, 1200)
        sc_h = st.slider("Screen Height (cm)", 5, 19, 12)
        sc_w = st.slider("Screen Width (cm)", 0, 18, 6)
        m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)

    with t3:
        blue = st.selectbox("Bluetooth", ["No", "Yes"])
        dual_sim = st.selectbox("Dual SIM", ["No", "Yes"])
        touch_screen = st.selectbox("Touch Screen", ["No", "Yes"], index=1)
        wifi = st.selectbox("WiFi", ["No", "Yes"], index=1)
        four_g = st.selectbox("4G Support", ["No", "Yes"], index=1)
        three_g = st.selectbox("3G Support", ["No", "Yes"], index=1)
        mobile_wt = st.selectbox("Weight Category", ["Low", "Med", "High"], index=1)

# ---------------------------------------------------------------------------
# Build the feature row exactly as encoded during training
# ---------------------------------------------------------------------------
yn_map = {"No": 0, "Yes": 1}
wt_map = {"Low": 0, "Med": 1, "High": 2}

input_dict = {
    "battery_power": battery_power, "blue": yn_map[blue], "clock_speed": clock_speed,
    "dual_sim": yn_map[dual_sim], "fc": fc, "four_g": yn_map[four_g],
    "int_memory": int_memory, "m_dep": m_dep, "mobile_wt": wt_map[mobile_wt],
    "n_cores": n_cores, "pc": pc, "px_height": px_height, "px_width": px_width,
    "ram": ram, "sc_h": sc_h, "sc_w": sc_w, "talk_time": talk_time,
    "three_g": yn_map[three_g], "touch_screen": yn_map[touch_screen], "wifi": yn_map[wifi],
}
input_df = pd.DataFrame([input_dict])[feature_cols]

# ---------------------------------------------------------------------------
# Main panel — everything fits in one screen
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">📱 Mobile Price Range Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Instant price-tier prediction from device specs, '
    'powered by a trained ML classifier.</div>', unsafe_allow_html=True
)

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<span class="model-chip">MODEL &nbsp;•&nbsp; {best_model_name}</span>', unsafe_allow_html=True)
    st.write("")
    predict_clicked = st.button("🔮  Predict Price Range")

    if predict_clicked:
        model_input = scaler.transform(input_df) if scaler is not None else input_df
        prediction = int(model.predict(model_input)[0])
        color = PRICE_COLORS[prediction]

        st.markdown(
            f'<div class="result-badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color}55;">PREDICTED TIER {prediction}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="price-value" style="color:{color};">{PRICE_LABELS[prediction]}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="price-label">Based on the specs you entered in the sidebar.</div>',
                    unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(model_input)[0]
            st.session_state["_last_proba"] = proba
    else:
        st.markdown(
            '<div class="price-label" style="margin-top:0.8rem;">'
            'Set the specs in the sidebar and click <b>Predict</b> to see the '
            'model\'s estimated price tier here.</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Class Probability Breakdown**")
    proba = st.session_state.get("_last_proba")
    if proba is not None:
        proba_df = pd.DataFrame({
            "Tier": [f"{i} · {PRICE_LABELS[i]}" for i in range(4)],
            "Probability": proba
        }).set_index("Tier")
        st.bar_chart(proba_df, height=260)
    else:
        st.caption("Probabilities will appear here after your first prediction.")
        st.write("")
        st.write("")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center;color:#5b6577;font-size:0.75rem;margin-top:1rem;">'
    'Introduction to Data Science — Mobile Price Prediction — BITS Pilani WILP</p>',
    unsafe_allow_html=True
)