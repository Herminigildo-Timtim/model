import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
from mtcn_model import TCNForecaster  # Import the correct trained model class

# Load scalers
try:
    scalers = joblib.load('scalers.pkl')
    numeric_scaler = scalers['numeric']
    input_size = numeric_scaler.n_features_in_  # Automatically match input size
except Exception as e:
    st.error(f"Error loading scalers: {e}")
    st.stop()

# Load the trained model
model = TCNForecaster(input_size=input_size, output_size=1, num_channels=[16, 32, 64])

# Load the model state
try:
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Time Series Prediction with TCN Model")
st.write("Enter values to predict temperature.")

# Feature inputs (ensure correct order and count)
st.subheader("Feature Inputs")

datetime_feature = st.number_input("Datetime Feature (hours since start)", value=0.0)
Tdew = st.number_input("Tdew (degC)", value=0.0)
rh = st.number_input("Relative Humidity (%)", value=0.0)
sh = st.number_input("Specific Humidity (g/kg)", value=0.0)
H2OC = st.number_input("H2OC (mmol/mol)", value=0.0)
rho = st.number_input("Density (g/m³)", value=0.0)

# Temporal features
day_of_week = (datetime_feature // 24) % 7  # Approximate day of the week
week = (datetime_feature // (24 * 7)) % 52  # Approximate week
month = (datetime_feature // (24 * 30)) % 12  # Approximate month

# Placeholder for missing features
num_missing_features = input_size - 9  # Adjust dynamically
extra_features = [st.number_input(f"Extra Feature {i+1}", value=0.0) for i in range(num_missing_features)]

# Prepare input
if st.button("Predict"):
    try:
        # Construct input array in correct order
        input_data = np.array([[Tdew, rh, sh, H2OC, rho, datetime_feature, day_of_week, week, month] + extra_features]).astype(np.float32)

        # Normalize input using the scaler
        input_data = numeric_scaler.transform(input_data)

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.write(f"Predicted Temperature (T degC): {prediction:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
