import os

import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor

import pickle
from typing import Literal
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn


model_name = os.getenv("MODEL_NAME", "blood_classifier_mobilenet_result.onnx")


def preprocess_pytorch_style(X):
    # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)  

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)


preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)



session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "basophil",
    "erythroblast",
    "monocyte",
    "myeloblast",
    "seg_neutrophil"
]


def predict_single(url):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event["url"]
    result = predict_single(url)
    return result



class Request(BaseModel):
    url: str = Field(..., example="https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/main/capstone-project/images/MYO_0042.jpg")

class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float

app = FastAPI(title="blood-model")

@app.post("/predict")
def predict(request: Request) -> PredictResponse:
    predictions = predict_single(request.url)

    top_class = max(predictions, key = predictions.get)
    top_probability = predictions[top_class]

    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_probability
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)