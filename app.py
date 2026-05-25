# ============================================================
# SIFCON STRENGTH PREDICTION WEB APPLICATION
# STREAMLIT CLOUD PERMANENT DEPLOYMENT VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ============================================================
# FIX RANDOMNESS
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIFCON Strength Predictor",
    layout="wide"
)

st.title("SIFCON Strength Prediction Using AI Models")

# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    # LOAD DATASET
    df = pd.read_csv("sifcon.csv")

    # FIX HEADERS
    df.columns = df.iloc[0]

    df = df[1:].reset_index(drop=True)

    # ========================================================
    # CONVERT NUMERIC
    # ========================================================

    numeric_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # ========================================================
    # REMOVE OUTLIERS
    # ========================================================

    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df['y'] >= lower) &
        (df['y'] <= upper)
    ]

    # ========================================================
    # LABEL ENCODING
    # ========================================================

    encoder = LabelEncoder()

    df['x7'] = encoder.fit_transform(df['x7'])

    # ========================================================
    # FEATURES & TARGET
    # ========================================================

    X = df.drop('y', axis=1)

    y = df['y']

    # ========================================================
    # SCALING
    # ========================================================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.15,
        random_state=42
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_model = XGBRegressor(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=4,

        min_child_weight=3,

        gamma=0.2,

        reg_alpha=0.5,

        reg_lambda=2,

        subsample=0.8,

        colsample_bytree=0.8,

        objective='reg:squarederror',

        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)

    # ========================================================
    # METRICS
    # ========================================================

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)

    return (
        rf_model,
        xgb_model,
        scaler,
        encoder,
        rf_mae,
        rf_rmse,
        rf_r2,
        xgb_mae,
        xgb_rmse,
        xgb_r2
    )

# ============================================================
# LOAD MODELS
# ============================================================

(
    rf_model,
    xgb_model,
    scaler,
    encoder,
    rf_mae,
    rf_rmse,
    rf_r2,
    xgb_mae,
    xgb_rmse,
    xgb_r2
) = train_models()

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

    rf_strength = rf_model.predict(sample_scaled)[0]

    xgb_strength = xgb_model.predict(sample_scaled)[0]

    # BEST MODEL
    if xgb_r2 >= rf_r2:
        best_model = "XGBoost"
        best_strength = xgb_strength
    else:
        best_model = "Random Forest"
        best_strength = rf_strength

    st.subheader("Predicted Strength Value")

    st.success(
        f"{best_model} Predicted Strength : "
        f"{best_strength:.2f} MPa"
    )

# ============================================================
# MODEL PERFORMANCE TABLE
# ============================================================

st.subheader("Model Performance")

results = pd.DataFrame({

    'Model': [
        'Random Forest',
        'XGBoost',
        'RNN',
        'LSTM'
    ],

    'MAE': [
        rf_mae,
        xgb_mae,
        rnn_mae,
        lstm_mae
    ],

    'RMSE': [
        rf_rmse,
        xgb_rmse,
        rnn_rmse,
        lstm_rmse
    ],

    'R2 Score': [
        rf_r2,
        xgb_r2,
        rnn_r2,
        lstm_r2
    ]
})

results = results.round(6)

st.dataframe(
    results,
    use_container_width=True
)
