# ============================================================
# SIFCON STRENGTH PREDICTION WEB APPLICATION
# STREAMLIT CLOUD FIXED VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# ============================================================
# FIX RANDOMNESS
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIFCON Strength Predictor",
    layout="wide"
)

st.title("SIFCON Strength Prediction Using AI Models")

# ============================================================
# LOAD & TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    # LOAD DATA
    df = pd.read_csv("sifcon.csv")

    # FIX HEADERS
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # CONVERT NUMERIC
    numeric_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # REMOVE OUTLIERS
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df['y'] >= lower) &
        (df['y'] <= upper)
    ]

    # LABEL ENCODING
    encoder = LabelEncoder()

    df['x7'] = encoder.fit_transform(df['x7'])

    # FEATURES
    X = df.drop('y', axis=1)
    y = df['y']

    # SCALING
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    rf_model.fit(X_scaled, y)

    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        objective='reg:squarederror',
        random_state=42
    )

    xgb_model.fit(X_scaled, y)

    return rf_model, xgb_model, scaler, encoder

# ============================================================
# LOAD MODELS
# ============================================================

rf_model, xgb_model, scaler, encoder = train_models()

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Enter Input Values")

x1 = st.sidebar.number_input(
    "Width of specimen (mm)",
    value=150.0
)

x2 = st.sidebar.number_input(
    "Length of specimen (mm)",
    value=150.0
)

x3 = st.sidebar.number_input(
    "Height of specimen (mm)",
    value=150.0
)

x4 = st.sidebar.number_input(
    "Weight of specimen (kg)",
    value=7.5
)

x5 = st.sidebar.number_input(
    "Steel fibres (%)",
    value=4.0
)

x6 = st.sidebar.number_input(
    "Curing days",
    value=7.0
)

test_type = st.sidebar.selectbox(
    "Test Type",
    ['c1', 's1', 'c2']
)

# ============================================================
# PREDICTION
# ============================================================

if st.sidebar.button("Predict Strength"):

    x7 = encoder.transform([test_type])[0]

    sample = [[
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    ]]

    sample_scaled = scaler.transform(sample)

    # BEST MODEL = XGBOOST
    strength = xgb_model.predict(sample_scaled)[0]

    st.subheader("Predicted Strength Value")

    st.success(
        f"Predicted Strength : {strength:.2f} MPa"
    )