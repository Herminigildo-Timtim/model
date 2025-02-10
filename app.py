import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
from mtcn_model import TCNForecaster  # Import the trained model class

# Load scalers
try:
    scalers = joblib.load('scalers.pkl')
    numeric_scaler = scalers['numeric']
    feature_names = scalers.get('feature_names', [])  # Retrieve feature names if available
except Exception as e:
    st.error(f"Error loading scalers: {e}")
    st.stop()

# Load model metadata (to get correct input size)
try:
    metadata = joblib.load('model_metadata.pkl')
    input_size = metadata['input_size']
    sequence_length = metadata['sequence_length']
except Exception as e:
    st.error(f"Error loading model metadata: {e}")
    st.stop()

# Initialize the model correctly
model = TCNForecaster(input_size=input_size, output_size=1, num_channels=[16, 32, 64])

# Load the state_dict properly
try:
    model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Time Series Temperature Prediction")
st.write("Enter values to predict temperature.")

# Auto-detect required features
if not feature_names:
    feature_names = ["Tdew (degC)", "rh (%)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m³)", "datetime_feature", "day_of_week", "week", "month"]

# Allow dynamic user input for detected features
st.subheader("Feature Inputs")
user_inputs = {}

for feature in feature_names:
    user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

# If model expects extra features, add them dynamically
num_missing_features = input_size - len(feature_names)

if num_missing_features > 0:
    for i in range(num_missing_features):
        extra_feature_name = f"Extra Feature {i+1}"
        user_inputs[extra_feature_name] = st.number_input(extra_feature_name, value=0.0)

# Convert inputs to NumPy array
if st.button("Predict"):
    try:
        # Convert dictionary values to list (preserve order)
        input_data = np.array([list(user_inputs.values())]).astype(np.float32)

        # Validate input shape before transforming
        if input_data.shape[1] != input_size:
            st.error(f"Feature mismatch! Expected {input_size} features, but got {input_data.shape[1]}")
            st.stop()

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
