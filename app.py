import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
from mtcn_model import TCNForecaster  # Import the correct trained model class

# Load scalers
try:
    scalers = joblib.load('scalers.pkl')
except Exception as e:
    st.error(f"Error loading scalers: {e}")

# Set correct input size (9 features based on training)
input_size = 9

# Load the trained model
model = TCNForecaster(input_size=input_size, output_size=1, num_channels=[16, 32, 64])

# Load the model state
try:
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI
st.title("Time Series Prediction with TCN Model")
st.write("Enter values to predict temperature.")

# Input fields for the correct 9 features
p_mbar = st.number_input("Pressure (mbar)")
Tdew = st.number_input("Tdew (degC)")
rh = st.number_input("Relative Humidity (%)")
VPact = st.number_input("VPact (mbar)")
H2OC = st.number_input("H2OC (mmol/mol)")
max_wv = st.number_input("Max Wind Velocity (m/s)")

# Add the missing 3 temporal features
datetime_feature = st.number_input("Datetime Feature (hours since start)")
day_of_week = (datetime_feature // 24) % 7  # Approximate day of the week
week = (datetime_feature // (24 * 7)) % 52  # Approximate week
month = (datetime_feature // (24 * 30)) % 12  # Approximate month

# Prepare input
if st.button("Predict"):
    try:
        # Ensure the input features match those used in training (9 features)
        input_data = np.array([[p_mbar, Tdew, rh, VPact, H2OC, max_wv, day_of_week, week, month]]).astype(np.float32)

        # Normalize input using the scalers
        input_data = scalers['numeric'].transform(input_data)

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.write(f"Predicted Temperature (T degC): {prediction:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
