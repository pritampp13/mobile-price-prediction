"""
Streamlit App - Mobile Price Range Prediction
Introduction to Data Science (S2-25_DSECLZG532) - BITS Pilani WILP

Run with:
    streamlit run app.py

This app loads the model artifacts produced by the accompanying Jupyter notebook
(Group2.ipynb) and lets a user enter new mobile phone specifications to get a
live prediction of the price range (0 = Low, 1 = Medium, 2 = High, 3 = Very High).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

st.set_page_config(page_title="Mobile Price Range Predictor", page_icon="📱", layout="centered")

st.title("📱 Mobile Price Range Predictor")
st.caption(f"Deployed model: **{best_model_name}**")
st.write(
    "Enter the technical specifications of a mobile phone below and click "
    "**Predict Price Range** to get an instant prediction from the trained model."
)

st.divider()

# ---------------------------------------------------------------------------
# Input widgets for every feature used during training
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1200)
    clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
    fc = st.slider("Front Camera (MP)", 0, 20, 5)
    int_memory = st.slider("Internal Memory (GB)", 2, 64, 32)
    m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)
    n_cores = st.slider("Number of Cores", 1, 8, 4)
    pc = st.slider("Primary Camera (MP)", 0, 20, 10)
    px_height = st.slider("Pixel Resolution Height", 0, 1960, 800)
    px_width = st.slider("Pixel Resolution Width", 500, 2000, 1200)
    ram = st.slider("RAM (MB)", 250, 4000, 2000)

with col2:
    sc_h = st.slider("Screen Height (cm)", 5, 19, 12)
    sc_w = st.slider("Screen Width (cm)", 0, 18, 6)
    talk_time = st.slider("Talk Time (hours)", 2, 20, 10)
    blue = st.selectbox("Bluetooth", ["No", "Yes"])
    dual_sim = st.selectbox("Dual SIM", ["No", "Yes"])
    touch_screen = st.selectbox("Touch Screen", ["No", "Yes"])
    wifi = st.selectbox("WiFi", ["No", "Yes"])
    mobile_wt = st.selectbox("Mobile Weight Category", ["Low", "Med", "High"])
    four_g = st.selectbox("4G Support", ["No", "Yes"])
    three_g = st.selectbox("3G Support", ["No", "Yes"])

# ---------------------------------------------------------------------------
# Build the feature row exactly as encoded during training
# ---------------------------------------------------------------------------
yn_map = {"No": 0, "Yes": 1}
wt_map = {"Low": 0, "Med": 1, "High": 2}

input_dict = {
    "battery_power": battery_power,
    "blue": yn_map[blue],
    "clock_speed": clock_speed,
    "dual_sim": yn_map[dual_sim],
    "fc": fc,
    "four_g": yn_map[four_g],
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": wt_map[mobile_wt],
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "three_g": yn_map[three_g],
    "touch_screen": yn_map[touch_screen],
    "wifi": yn_map[wifi],
}

input_df = pd.DataFrame([input_dict])[feature_cols]  # enforce correct column order

st.divider()

if st.button("🔍 Predict Price Range", type="primary"):
    model_input = scaler.transform(input_df) if scaler is not None else input_df
    prediction = model.predict(model_input)[0]

    st.success(f"### Predicted Price Range: **{prediction} — {PRICE_LABELS[prediction]}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(model_input)[0]
        proba_df = pd.DataFrame({
            "Price Range": [f"{i} ({PRICE_LABELS[i]})" for i in range(4)],
            "Probability": proba
        })
        st.bar_chart(proba_df.set_index("Price Range"))

st.divider()
st.caption("Introduction to Data Science — Mobile Price Prediction Assignment — BITS Pilani WILP")