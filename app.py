# ============================================================
# SIFCON STRENGTH PREDICTION WEB APPLICATION
# FINAL FULLY FIXED VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os

# ============================================================
# SKLEARN MODELS
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# ============================================================
# TENSORFLOW MODELS
# ============================================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    SimpleRNN,
    LSTM,
    Dropout
)

from tensorflow.keras.optimizers import Adam

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

st.markdown("---")

# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = pd.read_csv("sifcon.csv")

    # ========================================================
    # FIX COLUMN HEADERS
    # ========================================================

    df.columns = df.iloc[0]

    df = df[1:].reset_index(drop=True)

    # ========================================================
    # CONVERT NUMERIC
    # ========================================================

    numeric_cols = [
        'x1',
        'x2',
        'x3',
        'x4',
        'x5',
        'x6',
        'y'
    ]

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

    rf_mae = mean_absolute_error(y_test, rf_pred)

    rf_rmse = np.sqrt(
        mean_squared_error(y_test, rf_pred)
    )

    rf_r2 = r2_score(y_test, rf_pred)

    # ========================================================
    # IMPROVED XGBOOST MODEL
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

    xgb_mae = mean_absolute_error(y_test, xgb_pred)

    xgb_rmse = np.sqrt(
        mean_squared_error(y_test, xgb_pred)
    )

    xgb_r2 = r2_score(y_test, xgb_pred)

    # ========================================================
    # PREPARE DATA FOR RNN & LSTM
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
        verbose=0
    )

    rnn_pred = rnn_model.predict(
        X_test_rnn,
        verbose=0
    ).flatten()

    rnn_mae = mean_absolute_error(y_test, rnn_pred)

    rnn_rmse = np.sqrt(
        mean_squared_error(y_test, rnn_pred)
    )

    rnn_r2 = r2_score(y_test, rnn_pred)

    # ========================================================
    # LSTM MODEL
    # ========================================================

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
        verbose=0
    )

    lstm_pred = lstm_model.predict(
        X_test_rnn,
        verbose=0
    ).flatten()

    lstm_mae = mean_absolute_error(y_test, lstm_pred)

    lstm_rmse = np.sqrt(
        mean_squared_error(y_test, lstm_pred)
    )

    lstm_r2 = r2_score(y_test, lstm_pred)

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

        rf_mae,
        rf_rmse,
        rf_r2,

        xgb_mae,
        xgb_rmse,
        xgb_r2,

        rnn_mae,
        rnn_rmse,
        rnn_r2,

        lstm_mae,
        lstm_rmse,
        lstm_r2
    )

# ============================================================
# LOAD ALL MODELS
# ============================================================

(
    rf_model,
    xgb_model,
    rnn_model,
    lstm_model,
    scaler,
    encoder,

    rf_mae,
    rf_rmse,
    rf_r2,

    xgb_mae,
    xgb_rmse,
    xgb_r2,

    rnn_mae,
    rnn_rmse,
    rnn_r2,

    lstm_mae,
    lstm_rmse,
    lstm_r2

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
# PREDICTION BUTTON
# ============================================================

if st.sidebar.button("Predict Strength"):

    # ========================================================
    # PREPARE INPUT
    # ========================================================

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

    sample_rnn = sample_scaled.reshape(
        (1, 1, sample_scaled.shape[1])
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    rf_strength = rf_model.predict(sample_scaled)[0]

    xgb_strength = xgb_model.predict(sample_scaled)[0]

    rnn_strength = rnn_model.predict(
        sample_rnn,
        verbose=0
    ).flatten()[0]

    lstm_strength = lstm_model.predict(
        sample_rnn,
        verbose=0
    ).flatten()[0]

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.subheader("Predicted Strength Values")

    st.success(
        f"Random Forest Prediction : {rf_strength:.2f} MPa"
    )

    st.success(
        f"XGBoost Prediction : {xgb_strength:.2f} MPa"
    )

    st.success(
        f"RNN Prediction : {rnn_strength:.2f} MPa"
    )

    st.success(
        f"LSTM Prediction : {lstm_strength:.2f} MPa"
    )

    st.markdown("---")

    # ========================================================
    # MODEL PERFORMANCE TABLE
    # ========================================================

    st.subheader("Model Performance Comparison")

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

    st.dataframe(
        results.style.highlight_max(
            subset=['R2 Score'],
            color='lightgreen'
        ),
        use_container_width=True
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = results.loc[
        results['R2 Score'].idxmax()
    ]

    st.markdown("---")

    st.subheader("Best Performing Model")

    st.info(
        f"Best Model : {best_model['Model']}"
    )

    st.info(
        f"Best R² Score : {best_model['R2 Score']:.4f}"
    )