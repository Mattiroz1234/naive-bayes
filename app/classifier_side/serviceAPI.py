from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Dict
from .check import predict_naive_bayes
from .enums import AgeGroup, Smoker, AlcoholLevel, ExerciseLevel, DietQuality
import requests

app = FastAPI()

model_cache = {}



def get_model_cached(model_name: str):
    if model_name in model_cache:
        return model_cache[model_name]

    try:
        res = requests.get(f"http://trainer:8001/model?name={model_name}")
        # res = requests.get(f"http://localhost:8001/model?name={model_name}")
        if res.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found on trainer.")
        model = res.json()
        model_cache[model_name] = model
        return model
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Trainer service error: {str(e)}")


@app.get("/")
def root():
    return {"message": "Naive Bayes Prediction API"}


@app.get("/predict")
def predict(
    age_group: AgeGroup = Query(...),
    smoker: Smoker = Query(...),
    exercise: ExerciseLevel = Query(...),
    diet: DietQuality = Query(...),
    alcohol: AlcoholLevel = Query(...)
):
    input_data = {
        "age_group": age_group.value,
        "smoker": smoker.value,
        "exercise": exercise.value,
        "diet": diet.value,
        "alcohol": alcohol.value
    }

    model = get_model_cached("health")

    prediction = predict_naive_bayes(model, input_data)
    return {"prediction": prediction}


class GenericInput(BaseModel):
    model_name: str
    data: Dict[str, str]

@app.post("/predict/custom")
def gen_predict(input_data: GenericInput):
    model = get_model_cached(input_data.model_name)
    prediction = predict_naive_bayes(model, input_data.data)
    return {"prediction": prediction}


@app.post("/reload_model")
def reload_model(model_name: str):
    if model_name in model_cache:
        del model_cache[model_name]
    return {"message": f"Model '{model_name}' will be reloaded on next request."}
