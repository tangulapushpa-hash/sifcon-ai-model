# ============================================================
# SIFCON WEB APPLICATION - FINAL CLEAN VERSION
# ============================================================
# FEATURES:
# ✔ Fixed TensorFlow/Keras errors
# ✔ Same prediction values every run
# ✔ Only BEST MODEL prediction displayed
# ✔ No plots
# ✔ Faster execution
# ✔ Performance table included
# ✔ Stable Streamlit execution
# ============================================================

# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================
# pip install streamlit pandas numpy scikit-learn xgboost tensorflow
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
tensorflow-cpu==2.15.0
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# ============================================================
# FIX RANDOMNESS & TENSORFLOW ISSUES
# ============================================================

SEED = 42

np.random.seed(SEED)

random.seed(SEED)

tf.random.set_seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

tf.keras.backend.clear_session()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIFCON Strength Predictor",
    layout="wide"
)

st.title("SIFCON Strength Prediction Using AI Models")

# ============================================================
# TRAIN MODELS ONLY ONCE
# ============================================================

@st.cache_resource

def train_models():

    tf.keras.backend.clear_session()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    dataset_path = "sifcon.csv"

    df = pd.read_csv(dataset_path)

    # Fix headers
    df.columns = df.iloc[0]

    df = df[1:].reset_index(drop=True)

    # ========================================================
    # CONVERT DATA TYPES
    # ========================================================

    numeric_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # ========================================================
    # REMOVE OUTLIERS USING IQR
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
    # FEATURE SCALING
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
    # RANDOM FOREST MODEL
    # ========================================================

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    # ========================================================
    # XGBOOST MODEL
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
    # RESHAPE DATA FOR RNN/LSTM
    # ========================================================

    X_train_rnn = X_train.reshape(
        (X_train.shape[0], 1, X_train.shape[1])
    )

    X_test_rnn = X_test.reshape(
        (X_test.shape[0], 1, X_test.shape[1])
    )

    # ========================================================
    # RNN MODEL
    # ========================================================

    tf.keras.backend.clear_session()

    rnn_model = Sequential()

    rnn_model.add(
        SimpleRNN(
            64,
            activation='relu',
            input_shape=(1, X_train.shape[1])
        )
    )

    rnn_model.add(Dropout(0.2))

    rnn_model.add(Dense(32, activation='relu'))

    rnn_model.add(Dense(1))

    rnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )

    rnn_model.fit(
        X_train_rnn,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        shuffle=False,
        verbose=0
    )

    rnn_pred = np.array(
        rnn_model.predict(
            X_test_rnn,
            verbose=0
        )
    ).flatten()

    # ========================================================
    # LSTM MODEL
    # ========================================================

    tf.keras.backend.clear_session()

    lstm_model = Sequential()

    lstm_model.add(
        LSTM(
            64,
            activation='relu',
            input_shape=(1, X_train.shape[1])
        )
    )

    lstm_model.add(Dropout(0.2))

    lstm_model.add(Dense(32, activation='relu'))

    lstm_model.add(Dense(1))

    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )

    lstm_model.fit(
        X_train_rnn,
        y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        shuffle=False,
        verbose=0
    )

    lstm_pred = np.array(
        lstm_model.predict(
            X_test_rnn,
            verbose=0
        )
    ).flatten()

    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return (
        rf_model,
        xgb_model,
        rnn_model,
        lstm_model,
        scaler,
        encoder,
        y_test,
        rf_pred,
        xgb_pred,
        rnn_pred,
        lstm_pred
    )

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

(
    rf_model,
    xgb_model,
    rnn_model,
    lstm_model,
    scaler,
    encoder,
    y_test,
    rf_pred,
    xgb_pred,
    rnn_pred,
    lstm_pred
) = train_models()

# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

rnn_mae = mean_absolute_error(y_test, rnn_pred)
rnn_rmse = np.sqrt(mean_squared_error(y_test, rnn_pred))
rnn_r2 = r2_score(y_test, rnn_pred)

lstm_mae = mean_absolute_error(y_test, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)

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
# PREDICT BUTTON
# ============================================================

if st.sidebar.button("Predict Strength"):

    # Encode test type
    x7 = encoder.transform([test_type])[0]

    # Create input sample
    sample = [[
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    ]]

    # Scale input
    sample_scaled = scaler.transform(sample)

    # Reshape for RNN/LSTM
    sample_rnn = sample_scaled.reshape(
        (1, 1, sample_scaled.shape[1])
    )

    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    rf_strength = rf_model.predict(
        sample_scaled
    )[0]

    xgb_strength = xgb_model.predict(
        sample_scaled
    )[0]

    rnn_strength = np.array(
        rnn_model.predict(
            sample_rnn,
            verbose=0
        )
    ).flatten()[0]

    lstm_strength = np.array(
        lstm_model.predict(
            sample_rnn,
            verbose=0
        )
    ).flatten()[0]

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    scores = {
        "Random Forest": rf_r2,
        "XGBoost": xgb_r2,
        "RNN": rnn_r2,
        "LSTM": lstm_r2
    }

    predictions = {
        "Random Forest": rf_strength,
        "XGBoost": xgb_strength,
        "RNN": rnn_strength,
        "LSTM": lstm_strength
    }

    # Best model based on highest R² score
    best_model = max(scores, key=scores.get)

    best_strength = predictions[best_model]

    # ========================================================
    # DISPLAY BEST MODEL RESULT ONLY
    # ========================================================

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

# ============================================================
# END OF CODE
# ============================================================