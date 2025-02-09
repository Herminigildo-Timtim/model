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

# Get the correct input size (trained on 7, so use 7)
input_size = 7  # Change this to match the trained model's input size

# Load the trained model (Make sure this matches the training architecture)
model = TCNForecaster(input_size=input_size, output_size=1, num_channels=[16, 32, 64])  # Ensure input_size is 7

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

# Input fields based on selected features
datetime_feature = st.number_input("Datetime Feature (hours since start)")
Tdew = st.number_input("Tdew (degC)")
rh = st.number_input("Relative Humidity (%)")
sh = st.number_input("Specific Humidity (g/kg)")
H2OC = st.number_input("H2OC (mmol/mol)")
rho = st.number_input("Density (g/m³)")

# Add an extra input field if necessary
extra_feature = st.number_input("Extra Feature (if needed for input size match)", value=0.0)

# Prepare input
if st.button("Predict"):
    try:
        input_data = np.array([[Tdew, rh, sh, H2OC, rho, datetime_feature, extra_feature]]).astype(np.float32)

        # Normalize using the scalers
        input_data[:, :-1] = scalers['numeric'].transform(input_data[:, :-1])

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.write(f"Predicted Temperature (T degC): {prediction:.2f}")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
