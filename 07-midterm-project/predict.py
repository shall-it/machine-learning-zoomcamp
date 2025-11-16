#!/usr/bin/env python
# coding: utf-8


import pickle
import uvicorn
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field
from pydantic import ConfigDict

import xgboost as xgb


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    age: int
    gender: str
    region: str
    income_level: str
    education_level: str
    daily_role: str
    device_hours_per_day: float
    phone_unlocks: int
    notifications_per_day: int
    social_media_mins: int
    study_mins: int
    physical_activity_days: float
    sleep_hours: float
    sleep_quality: float
    anxiety_score: float
    depression_score: float
    stress_level: float
    happiness_score: float
    focus_score: float
    device_type: str
    productivity_score: float
    digital_dependence_score: float


# response
class PredictResponse(BaseModel):
    risk_probability: float
    risk: bool


app = FastAPI(title="person-risk-prediction")


with open ('model.bin', 'rb') as f_in:
    # pipeline = pickle.load(f_in)
    model, dv = pickle.load(f_in)


def predict_single(person):
    X = dv.transform([person])
    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    result = model.predict(dmatrix)[0]
    return float(result)



@app.post("/predict")
def predict(person: Person) -> PredictResponse:
    prob = predict_single(person.model_dump())

    return PredictResponse(
        risk_probability=prob,
        risk=bool(prob >= 0.5)
    )


@app.get("/health")
def health():
    return "HEALTH"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


# uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload 
