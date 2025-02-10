import streamlit as st
import torch
import numpy as np
import joblib
import datetime  # For extracting time-based features
from sklearn.preprocessing import MinMaxScaler
from mtcn_model import TCNForecaster
import traceback  # To log detailed errors

# Enable Debug Mode
DEBUG_MODE = True

# Load trained model and feature selection details
model_path = "trained_tcn_model.pt"
feature_info_path = "feature_selection_info.pkl"
scaler_path = "scalers.pkl"  # Path to pre-trained scaler

# Load feature selection information
try:
    feature_info = joblib.load(feature_info_path)
    assert "selected_features" in feature_info, "Key 'selected_features' not found in feature info."
    assert "sequence_length" in feature_info, "Key 'sequence_length' not found in feature info."

    selected_features = feature_info["selected_features"]
    sequence_length = feature_info["sequence_length"]

    # Remove date-related features from manual inputs (since we'll extract them from date picker)
    selected_features = [
        feat for feat in selected_features if feat not in ["datetime_feature", "day_of_week", "week", "month"]
    ]

    if DEBUG_MODE:
        st.write("✅ Feature info loaded successfully.")
        st.write(f"Selected Features: {selected_features}")
        st.write(f"Sequence Length: {sequence_length}")

except Exception as e:
    st.error(f"❌ Error loading feature info: {e}")
    st.write(traceback.format_exc())  # Print the full error traceback
    st.stop()

# Load trained MinMaxScaler
try:
    scalers = joblib.load(scaler_path)  # This loads a dictionary
    if "numeric" in scalers and isinstance(scalers["numeric"], MinMaxScaler):
        numeric_scaler = scalers["numeric"]  # ✅ Extract the MinMaxScaler
        st.write("✅ Numeric MinMaxScaler loaded successfully.")
    else:
        raise ValueError("❌ Numeric MinMaxScaler not found in scaler dictionary.")
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    st.write(traceback.format_exc())
    st.stop()


# Load trained model
def load_model(input_size, output_size=1, num_channels=[16, 32, 64]):
    try:
        model = TCNForecaster(input_size, output_size, num_channels)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        if DEBUG_MODE:
            st.write("✅ Model loaded successfully.")

        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.write(traceback.format_exc())  # Print the full error traceback
        st.stop()


# Streamlit UI
st.title("Time Series Forecasting App (Debug Mode)")
st.write("Select a date and enter feature values to predict temperature.")

# Date Picker for automatic extraction of time-based features
st.subheader("Select Date")
selected_date = st.date_input("Choose a Date", datetime.date.today())

# Extract time-based features from the selected date
day_of_week = selected_date.weekday()  # Monday=0, Sunday=6
week_number = selected_date.isocalendar()[1]  # ISO week number
month = selected_date.month  # Month number (1-12)

# Calculate datetime_feature (Example: Normalization based on date values)
datetime_feature = (day_of_week + week_number + month) / 100  

# Debugging: Show extracted time-based features
if DEBUG_MODE:
    st.subheader("Debugging: Extracted Temporal Features")
    st.write(f"Selected Date: {selected_date}")
    st.write(f"Day of Week: {day_of_week}, Week Number: {week_number}, Month: {month}")
    st.write(f"Calculated datetime_feature: {datetime_feature}")

# Create dynamic input fields for numerical features (excluding date-related ones)
st.subheader("Enter Feature Values")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

# Predict Button
if st.button("Predict Temperature"):
    try:
        # ✅ Get the expected feature names from `feature_selection_info.pkl`
        expected_features = feature_info["selected_features"]
        expected_features = [
            feat for feat in expected_features if feat not in ["datetime_feature", "day_of_week", "week", "month"]
        ]  # Remove manual date inputs
        expected_features.extend(["datetime_feature", "day_of_week", "week", "month"])  # ✅ Append time-based features

        expected_feature_count = len(expected_features)  # ✅ Feature count includes time-based features

        # ✅ Ensure MinMaxScaler has the correct feature count
        scaler_feature_count = len(scalers["feature_names"])  # Should match expected feature count

        # Convert input to model format
        input_values = [user_input.get(feat, 0.0) for feat in selected_features]
        input_values.extend([datetime_feature, day_of_week, week_number, month])  # ✅ Append calculated time features

        # Convert to NumPy array & reshape
        input_values = np.array(input_values).reshape(1, -1)

        # Debugging: Check feature counts
        if DEBUG_MODE:
            st.subheader("Debugging: Feature Count Check")
            st.write(f"Expected Feature Count (from `feature_selection_info.pkl`): {expected_feature_count}")
            st.write(f"Scaler Expected Feature Count: {scaler_feature_count}")
            st.write(f"Actual Feature Count Used: {input_values.shape[1]}")

        # ✅ Ensure input feature count is correct before scaling
        if input_values.shape[1] != scaler_feature_count:
            st.error(f"❌ Feature count mismatch! Expected {scaler_feature_count}, but got {input_values.shape[1]}.")
            st.write("Expected Feature Order:", scalers["feature_names"])
            st.write("Actual Feature Order Used:", expected_features)
            st.stop()

        # Scale input using the corrected MinMaxScaler
        try:
            scaled_input = numeric_scaler.transform(input_values)

            # Debugging: Scaled input
            if DEBUG_MODE:
                st.subheader("Debugging: Scaled Inputs")
                st.write("Scaled Input:", scaled_input.tolist())

        except Exception as e:
            st.error(f"❌ Scaling error: {e}")
            st.write(traceback.format_exc())  # Show full error traceback
            st.stop()

        # Convert to tensor
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)  # [batch_size, seq_length, num_features]

        # Debugging Print: Tensor shape
        if DEBUG_MODE:
            st.subheader("Debugging: Final Input Tensor")
            st.write(f"Tensor Shape: {input_tensor.shape}")

        # Ensure input tensor shape is correct
        assert input_tensor.shape[2] == expected_feature_count, f"Expected input size {expected_feature_count}, but got {input_tensor.shape[2]}."

        # Load model and predict
        model = load_model(expected_feature_count)

        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().flatten()

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Date: {selected_date.strftime('%Y-%m-%d')}**")
        st.write(f"**Predicted Temperature (T in degC): {prediction[0]:.2f}**")

        # Debugging: Check model output
        if DEBUG_MODE:
            st.subheader("Debugging: Model Output")
            st.write(f"Raw Model Output: {prediction}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        st.write(traceback.format_exc())  # Print full error traceback
