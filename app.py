import streamlit as st
import joblib
import torch
import numpy as np
from mtcn_model import TCNForecaster  # Import the trained model class

# Load scalers and check expected input size
try:
    scalers = joblib.load('scalers.pkl')
    expected_features = scalers.get('feature_count', 9)  # Ensure we match training
except Exception as e:
    st.error(f"Error loading scalers: {e}")
    expected_features = 9

# Set input size dynamically
input_size = expected_features

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

# Input fields for the correct features
p_mbar = st.number_input("Pressure (mbar)")
Tdew = st.number_input("Tdew (degC)")
rh = st.number_input("Relative Humidity (%)")
VPact = st.number_input("VPact (mbar)")
H2OC = st.number_input("H2OC (mmol/mol)")
max_wv = st.number_input("Max Wind Velocity (m/s)")
datetime_feature = st.number_input("Datetime Feature (hours since start)")

# Compute the missing temporal features
day_of_week = (datetime_feature // 24) % 7
week = (datetime_feature // (24 * 7)) % 52
month = (datetime_feature // (24 * 30)) % 12

# Prepare input
if st.button("Predict"):
    try:
        input_data = np.array([[p_mbar, Tdew, rh, VPact, H2OC, max_wv, day_of_week, week, month]]).astype(np.float32)

        # Check feature count
        if input_data.shape[1] != expected_features:
            st.error(f"Feature count mismatch: Expected {expected_features}, but got {input_data.shape[1]}")
        else:
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
