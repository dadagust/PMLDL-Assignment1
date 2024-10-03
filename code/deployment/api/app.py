from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests

app = FastAPI()

model = joblib.load('models/model.pkl')


class InputData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

@app.post("/predict/")
def predict(input_data: InputData):
    # Prepare the input features as a numpy array
    input_features = np.array([[input_data.mean_radius, input_data.mean_texture, input_data.mean_perimeter,
                                input_data.mean_area, input_data.mean_smoothness, input_data.mean_compactness,
                                input_data.mean_concavity, input_data.mean_concave_points, input_data.mean_symmetry,
                                input_data.mean_fractal_dimension, input_data.radius_error, input_data.texture_error,
                                input_data.perimeter_error, input_data.area_error, input_data.smoothness_error,
                                input_data.compactness_error, input_data.concavity_error, input_data.concave_points_error,
                                input_data.symmetry_error, input_data.fractal_dimension_error, input_data.worst_radius,
                                input_data.worst_texture, input_data.worst_perimeter, input_data.worst_area,
                                input_data.worst_smoothness, input_data.worst_compactness, input_data.worst_concavity,
                                input_data.worst_concave_points, input_data.worst_symmetry, input_data.worst_fractal_dimension]])

    # Make the prediction using the loaded model
    prediction = model.predict(input_features)

    # Return the prediction as a response
    return {"prediction": int(prediction[0])}