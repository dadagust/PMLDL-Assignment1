import streamlit as st
import requests

# Define the FastAPI endpoint
API_URL = "http://api:8000/predict/"

# Input fields
st.title("Machine Learning Model Prediction")
mean_radius = st.number_input("mean_radius", min_value=0.0, step=0.01)
mean_texture = st.number_input("mean_texture", min_value=0.0, step=0.01)
mean_perimeter = st.number_input("mean_perimeter", min_value=0.0, step=0.01)
mean_area = st.number_input("mean_area", min_value=0.0, step=1.0)
mean_smoothness = st.number_input("mean_smoothness", min_value=0.0, step=0.001)
mean_compactness = st.number_input("mean_compactness", min_value=0.0, step=0.001)
mean_concavity = st.number_input("mean_concavity", min_value=0.0, step=0.001)
mean_concave_points = st.number_input("mean_concave_points", min_value=0.0, step=0.001)
mean_symmetry = st.number_input("mean_symmetry", min_value=0.0, step=0.001)
mean_fractal_dimension = st.number_input("mean_fractal_dimension", min_value=0.0, step=0.001)

radius_error = st.number_input("radius_error", min_value=0.0, step=0.01)
texture_error = st.number_input("texture_error", min_value=0.0, step=0.01)
perimeter_error = st.number_input("perimeter_error", min_value=0.0, step=0.1)
area_error = st.number_input("area_error", min_value=0.0, step=1.0)
smoothness_error = st.number_input("smoothness_error", min_value=0.0, step=0.001)
compactness_error = st.number_input("compactness_error", min_value=0.0, step=0.001)
concavity_error = st.number_input("concavity_error", min_value=0.0, step=0.001)
concave_points_error = st.number_input("concave_points_error", min_value=0.0, step=0.001)
symmetry_error = st.number_input("symmetry_error", min_value=0.0, step=0.001)
fractal_dimension_error = st.number_input("fractal_dimension_error", min_value=0.0, step=0.001)

worst_radius = st.number_input("worst_radius", min_value=0.0, step=0.01)
worst_texture = st.number_input("worst_texture", min_value=0.0, step=0.01)
worst_perimeter = st.number_input("worst_perimeter", min_value=0.0, step=0.1)
worst_area = st.number_input("worst_area", min_value=0.0, step=1.0)
worst_smoothness = st.number_input("worst_smoothness", min_value=0.0, step=0.001)
worst_compactness = st.number_input("worst_compactness", min_value=0.0, step=0.001)
worst_concavity = st.number_input("worst_concavity", min_value=0.0, step=0.001)
worst_concave_points = st.number_input("worst_concave_points", min_value=0.0, step=0.001)
worst_symmetry = st.number_input("worst_symmetry", min_value=0.0, step=0.001)
worst_fractal_dimension = st.number_input("worst_fractal_dimension", min_value=0.0, step=0.001)

# Button to trigger prediction
if st.button("Make Prediction"):
    input_data = {
        "mean_radius": mean_radius,
        "mean_texture": mean_texture,
        "mean_perimeter": mean_perimeter,
        "mean_area": mean_area,
        "mean_smoothness": mean_smoothness,
        "mean_compactness": mean_compactness,
        "mean_concavity": mean_concavity,
        "mean_concave_points": mean_concave_points,
        "mean_symmetry": mean_symmetry,
        "mean_fractal_dimension": mean_fractal_dimension,
        "radius_error": radius_error,
        "texture_error": texture_error,
        "perimeter_error": perimeter_error,
        "area_error": area_error,
        "smoothness_error": smoothness_error,
        "compactness_error": compactness_error,
        "concavity_error": concavity_error,
        "concave_points_error": concave_points_error,
        "symmetry_error": symmetry_error,
        "fractal_dimension_error": fractal_dimension_error,
        "worst_radius": worst_radius,
        "worst_texture": worst_texture,
        "worst_perimeter": worst_perimeter,
        "worst_area": worst_area,
        "worst_smoothness": worst_smoothness,
        "worst_compactness": worst_compactness,
        "worst_concavity": worst_concavity,
        "worst_concave_points": worst_concave_points,
        "worst_symmetry": worst_symmetry,
        "worst_fractal_dimension": worst_fractal_dimension
    }

    # Make the POST request to the FastAPI server
    response = requests.post(API_URL, json=input_data)

    # Display the prediction result
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Error occurred during prediction!")
