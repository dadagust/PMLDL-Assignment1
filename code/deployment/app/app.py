import streamlit as st
import requests

# Define the FastAPI endpoint
API_URL = "http://api:8000/predict/"

# Input fields
st.title("Machine Learning Model Prediction")
feature1 = st.number_input("Feature 1", min_value=0.0, step=0.01)
feature2 = st.number_input("Feature 2", min_value=0.0, step=0.01)
feature3 = st.number_input("Feature 3", min_value=0.0, step=0.01)
feature4 = st.number_input("Feature 4", min_value=0.0, step=0.01)
feature5 = st.number_input("Feature 5", min_value=0.0, step=0.01)

# Button to trigger prediction
if st.button("Make Prediction"):
    input_data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4,
        "feature5": feature5
    }
    response = requests.post(API_URL, json=input_data)
    prediction = response.json()["prediction"]
    st.write(f"Prediction: {prediction}")

