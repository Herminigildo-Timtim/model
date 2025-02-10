import streamlit as st
import torch
import numpy as np
import joblib
import datetime
from sklearn.preprocessing import MinMaxScaler
from mtcn_model import TCNForecaster

# Load trained model and feature selection details
model_path = "trained_tcn_model.pt"
feature_info_path = "feature_selection_info.pkl"

# Load feature selection information
feature_info = joblib.load(feature_info_path)
selected_features = feature_info["selected_features"]
sequence_length = feature_info["sequence_length"]

# Define scaler
scaler = MinMaxScaler()

# Load trained model
def load_model(input_size, output_size=1, num_channels=[16, 32, 64]):
    model = TCNForecaster(input_size, output_size, num_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Streamlit UI
st.title("Time Series Forecasting App")
st.write("Enter feature values manually along with the date to predict temperature.")

# Date Input (No Time Needed)
st.subheader("Select Date")
user_date = st.date_input("Choose a Date", datetime.date.today())

# Extract temporal features
day_of_week = user_date.weekday()  # Monday=0, Sunday=6
week_number = user_date.isocalendar()[1]
month = user_date.month

# Create dynamic input fields for numerical features
st.subheader("Enter Feature Values")
user_input = {}
for feature in selected_features:
    if feature not in ["day_of_week", "week", "month"]:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

# Predict Button
if st.button("Predict Temperature"):
    # Convert input to model format
    input_values = np.array([user_input[feat] for feat in selected_features if feat not in ["day_of_week", "week", "month"]])

    # Append extracted temporal features
    input_values = np.append(input_values, [day_of_week, week_number, month]).reshape(1, -1)

    # Scale input
    scaled_input = scaler.fit_transform(input_values)

    # Convert to tensor
    input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)  # [batch_size, seq_length, num_features]

    # Load model and predict
    model = load_model(len(selected_features))
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy().flatten()

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Date: {user_date.strftime('%Y-%m-%d')}**")
    st.write(f"**Predicted Temperature (T in degC): {prediction[0]:.2f}**")
