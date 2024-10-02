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

@app.post("/predict/")
def predict(input_data: InputData):
    input_features = np.array(
        [[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4, input_data.feature5]])

    prediction = model.predict(input_features)

    return {"prediction": int(prediction[0])}
