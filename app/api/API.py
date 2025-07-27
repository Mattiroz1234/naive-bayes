from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from core.training import train_naive_bayes
from core.check import predict_naive_bayes
from .enums import AgeGroup, Smoker, AlcoholLevel, ExerciseLevel, DietQuality
import os

app = FastAPI()

# Creates a model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "health_generated.csv")
df_health = pd.read_csv(csv_path)
target_health = "risk"
model_health = train_naive_bayes(df_health, target_health)

models = {
    "health": {
        "model": model_health,
        "target": target_health
    }
}


# bace function
@app.get("/")
def root():
    return {"message": "Naive Bayes Prediction API"}


# get function for health model
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

    prediction = predict_naive_bayes(model_health, input_data)
    return {"prediction": prediction}



class GenericInput(BaseModel):
    model_name: str
    data: Dict[str, str]

# General function for all models
@app.post("/predict/custom")
def gen_predict(input_data: GenericInput):
    if input_data.model_name not in models:
        return {"error": f"Model '{input_data.model_name}' not found."}

    model_info = models[input_data.model_name]
    prediction = predict_naive_bayes(model_info["model"], input_data.data)
    return {"prediction": prediction}
