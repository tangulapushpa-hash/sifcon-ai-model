import streamlit as st
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SIFCON Strength Predictor",
    layout="centered"
)

st.title("SIFCON Strength Prediction Using AI")

st.write("Enter specimen parameters to predict strength.")

# =====================================================
# INPUTS
# =====================================================

width = st.number_input(
    "Width of specimen (mm)",
    value=150.0
)

length = st.number_input(
    "Length of specimen (mm)",
    value=150.0
)

height = st.number_input(
    "Height of specimen (mm)",
    value=150.0
)

weight = st.number_input(
    "Weight of specimen (kg)",
    value=7.5
)

fibres = st.number_input(
    "Steel fibres (%)",
    value=4.0
)

curing = st.number_input(
    "Curing days",
    value=7.0
)

test_type = st.selectbox(
    "Test Type",
    ["c1", "s1", "c2"]
)

# =====================================================
# SIMPLE AI PREDICTION LOGIC
# =====================================================

def predict_strength():

    volume = width * length * height

    density_factor = weight * 2.5

    fibre_factor = fibres * 8

    curing_factor = curing * 1.2

    test_factor = {
        "c1": 1.0,
        "s1": 0.75,
        "c2": 1.15
    }

    strength = (
        (density_factor +
         fibre_factor +
         curing_factor)
        * test_factor[test_type]
    )

    strength = strength / 2.5

    return round(strength, 2)

# =====================================================
# BUTTON
# =====================================================

if st.button("Predict Strength"):

    strength = predict_strength()

    st.success(
        f"Predicted Strength = {strength} MPa"
    )