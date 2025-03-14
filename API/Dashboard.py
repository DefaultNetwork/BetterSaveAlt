import streamlit as st
import requests
from datetime import datetime

st.title("Energy Forecasting Dashboard")
st.write("This dashboard allows you to predict energy consumption and get storage recommendations.")

# Section for Energy Forecast Prediction
st.header("Energy Consumption Prediction")
with st.form("predict_form"):
    # Input fields for prediction
    date_input = st.date_input("Select Date", datetime.today())
    energy_generation = st.number_input("Enter Energy Generation", min_value=0, value=150)
    submitted_predict = st.form_submit_button("Predict Energy Consumption")

    if submitted_predict:
        # Prepare the payload for the /predict endpoint
        payload = {
            "date": date_input.strftime("%Y-%m-%d"),
            "energy_generation": energy_generation
        }
        # Make a POST request to the Flask API
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json=payload)
            result = response.json()
            if "predicted_energy" in result:
                st.success(f"Predicted Energy Consumption: {result['predicted_energy']}")
            else:
                st.error("Error in prediction: " + result.get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Request failed: {e}")

# Section for Storage Allocation Recommendation
st.header("Storage Allocation Recommendation")
with st.form("allocate_form"):
    # Input field for storage recommendation
    predicted_energy_input = st.number_input("Enter Predicted Energy Consumption", min_value=0, value=150)
    submitted_allocate = st.form_submit_button("Get Storage Recommendation")

    if submitted_allocate:
        # Prepare the payload for the /allocate endpoint
        payload = {
            "predicted_energy": predicted_energy_input
        }
        # Make a POST request to the Flask API
        try:
            response = requests.post("http://127.0.0.1:5000/allocate", json=payload)
            result = response.json()
            if "recommended_storage" in result:
                st.success(f"Recommended Storage: {result['recommended_storage']}")
            else:
                st.error("Error in allocation: " + result.get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Request failed: {e}")
