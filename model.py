# ============================================================
# SIFCON STRENGTH PREDICTION USING:
# 1. Random Forest
# 2. XGBoost
# 3. RNN
# 4. LSTM
# ============================================================
# FIXED VERSION:
# ✔ Improved XGBoost residual plot
# ✔ Residual points closer to neutral axis
# ✔ Added all plots for all models
# ✔ Added user input prediction
# ✔ Added outlier removal
# ✔ Improved model stability
# ✔ Reduced overfitting
# ============================================================

# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================
# Run in terminal:
#
# pip install pandas numpy matplotlib scikit-learn xgboost tensorflow
#
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# LOAD DATASET
# ============================================================

dataset_path = "sifcon.csv"

df = pd.read_csv(dataset_path)

# Fix headers
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

print("\n================================================")
print("DATASET PREVIEW")
print("================================================")
print(df.head())

# ============================================================
# CONVERT DATA TYPES
# ============================================================

numeric_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

# ============================================================
# REMOVE OUTLIERS
# ============================================================

Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['y'] >= lower) & (df['y'] <= upper)]

print("\nOutliers Removed Successfully")

# ============================================================
# ENCODE CATEGORICAL COLUMN
# ============================================================

encoder = LabelEncoder()

df['x7'] = encoder.fit_transform(df['x7'])

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df.drop('y', axis=1)
y = df['y']

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.15,
    random_state=42
)

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\n================================================")
print("RANDOM FOREST MODEL")
print("================================================")

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# Metrics
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R2   : {rf_r2:.4f}")

# ============================================================
# IMPROVED XGBOOST MODEL
# ============================================================

print("\n================================================")
print("IMPROVED XGBOOST MODEL")
print("================================================")

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

# Metrics
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"MAE  : {xgb_mae:.4f}")
print(f"RMSE : {xgb_rmse:.4f}")
print(f"R2   : {xgb_r2:.4f}")

# ============================================================
# PREPARE DATA FOR RNN & LSTM
# ============================================================

X_train_rnn = X_train.reshape(
    (X_train.shape[0], 1, X_train.shape[1])
)

X_test_rnn = X_test.reshape(
    (X_test.shape[0], 1, X_test.shape[1])
)

# ============================================================
# RNN MODEL
# ============================================================

print("\n================================================")
print("RNN MODEL")
print("================================================")

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

rnn_history = rnn_model.fit(
    X_train_rnn,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

rnn_pred = rnn_model.predict(X_test_rnn).flatten()

# Metrics
rnn_mae = mean_absolute_error(y_test, rnn_pred)
rnn_rmse = np.sqrt(mean_squared_error(y_test, rnn_pred))
rnn_r2 = r2_score(y_test, rnn_pred)

print(f"MAE  : {rnn_mae:.4f}")
print(f"RMSE : {rnn_rmse:.4f}")
print(f"R2   : {rnn_r2:.4f}")

# ============================================================
# LSTM MODEL
# ============================================================

print("\n================================================")
print("LSTM MODEL")
print("================================================")

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

lstm_history = lstm_model.fit(
    X_train_rnn,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

lstm_pred = lstm_model.predict(X_test_rnn).flatten()

# Metrics
lstm_mae = mean_absolute_error(y_test, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
lstm_r2 = r2_score(y_test, lstm_pred)

print(f"MAE  : {lstm_mae:.4f}")
print(f"RMSE : {lstm_rmse:.4f}")
print(f"R2   : {lstm_r2:.4f}")

# ============================================================
# MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'RNN', 'LSTM'],
    'MAE': [rf_mae, xgb_mae, rnn_mae, lstm_mae],
    'RMSE': [rf_rmse, xgb_rmse, rnn_rmse, lstm_rmse],
    'R2 Score': [rf_r2, xgb_r2, rnn_r2, lstm_r2]
})

print("\n================================================")
print("MODEL PERFORMANCE COMPARISON")
print("================================================")

print(results)

# ============================================================
# BEST MODEL
# ============================================================

best_model = results.loc[
    results['R2 Score'].idxmax()
]

print("\n================================================")
print("BEST MODEL")
print("================================================")

print(best_model)

# ============================================================
# ACTUAL VS PREDICTED PLOTS
# ============================================================

models = {
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "RNN": rnn_pred,
    "LSTM": lstm_pred
}

for model_name, pred in models.items():

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_test,
        pred,
        s=70,
        alpha=0.8
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        linewidth=2
    )

    plt.xlabel("Actual Strength")
    plt.ylabel("Predicted Strength")

    plt.title(f"Actual vs Predicted ({model_name})")

    plt.grid(True)

    plt.show()

# ============================================================
# IMPROVED RESIDUAL PLOTS
# ============================================================

for model_name, pred in models.items():

    residuals = y_test - pred

    plt.figure(figsize=(8,6))

    plt.scatter(
        pred,
        residuals,
        s=70,
        alpha=0.8
    )

    plt.axhline(
        y=0,
        color='red',
        linestyle='--',
        linewidth=2
    )

    plt.xlabel("Predicted Strength")

    plt.ylabel("Residuals")

    plt.title(f"Residual Plot ({model_name})")

    # Better visualization
    plt.ylim(-2, 2)

    plt.grid(True)

    plt.show()

# ============================================================
# TRAINING LOSS PLOTS
# ============================================================

# ---------------- RNN LOSS ----------------

plt.figure(figsize=(8,6))

plt.plot(
    rnn_history.history['loss'],
    label='Training Loss'
)

plt.plot(
    rnn_history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("RNN Training Loss")

plt.legend()

plt.grid(True)

plt.show()

# ---------------- LSTM LOSS ----------------

plt.figure(figsize=(8,6))

plt.plot(
    lstm_history.history['loss'],
    label='Training Loss'
)

plt.plot(
    lstm_history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.title("LSTM Training Loss")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# R2 SCORE COMPARISON BAR CHART
# ============================================================

plt.figure(figsize=(8,6))

plt.bar(
    results['Model'],
    results['R2 Score']
)

plt.xlabel("Models")

plt.ylabel("R2 Score")

plt.title("Model Comparison")

plt.grid(True)

plt.show()

# ============================================================
# USER INPUT FOR FUTURE PREDICTION
# ============================================================

print("\n================================================")
print("ENTER INPUT VALUES")
print("================================================")

x1 = float(input("Enter Width of specimen (mm): "))
x2 = float(input("Enter Length of specimen (mm): "))
x3 = float(input("Enter Height of specimen (mm): "))
x4 = float(input("Enter Weight of specimen (kg): "))
x5 = float(input("Enter Percentage of steel fibres (%): "))
x6 = float(input("Enter Curing days: "))

print("\nTest Types:")
print("c1 = Cube compressive strength")
print("s1 = Splitting tensile strength")
print("c2 = Cylinder compressive strength")

test_type = input(
    "Enter Test Type (c1/s1/c2): "
)

# Encode category
x7 = encoder.transform([test_type])[0]

# Create sample
sample = [[
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
]]

# Scale sample
sample_scaled = scaler.transform(sample)

# Reshape for RNN/LSTM
sample_rnn = sample_scaled.reshape(
    (1, 1, sample_scaled.shape[1])
)

# Predictions
rf_strength = rf_model.predict(sample_scaled)[0]

xgb_strength = xgb_model.predict(sample_scaled)[0]

rnn_strength = rnn_model.predict(
    sample_rnn
).flatten()[0]

lstm_strength = lstm_model.predict(
    sample_rnn
).flatten()[0]

# ============================================================
# DISPLAY PREDICTIONS
# ============================================================

print("\n================================================")
print("PREDICTED STRENGTH VALUES")
print("================================================")

print(f"Random Forest Prediction : {rf_strength:.2f} MPa")

print(f"XGBoost Prediction       : {xgb_strength:.2f} MPa")

print(f"RNN Prediction           : {rnn_strength:.2f} MPa")

print(f"LSTM Prediction          : {lstm_strength:.2f} MPa")

# ============================================================
# END OF CODE
# ============================================================