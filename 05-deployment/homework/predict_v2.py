import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from pydantic import ConfigDict

class Client(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


app = FastAPI(title="scoring")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(client: dict):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)

@app.post("/predict")
def predict(client: Client):
    prob = predict_single(client.dict())
    return {
        "score_probability": prob,
        "score": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


# uv run uvicorn predict_v2:app --host 0.0.0.0 --port 9696 --reload 