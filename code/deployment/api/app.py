from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests

app = FastAPI()

model = joblib.load('models/model.pkl')


class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float


@app.get("/")
def read_root():
    url = "http://localhost:8000/predict/"
    data = {
        "feature1": 1.0,
        "feature2": 4.5,
        "feature3": 2.0,
        "feature4": 0.1,
        "feature5": 3.0
    }
    response = requests.post(url, json=data)
    print(response.json())
    return {"message": "Model Predicted:", "prediction": response.json()}


@app.post("/predict/")
def predict(input_data: InputData):
    input_features = np.array(
        [[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4, input_data.feature5]])

    prediction = model.predict(input_features)

    return {"prediction": int(prediction[0])}
