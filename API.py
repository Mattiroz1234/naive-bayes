from fastapi import FastAPI, Query
from training import train_naive_bayes
from  check import predict_naive_bayes
import pandas as pd
from enums import AgeGroup, Smoker, AlcoholLevel, ExerciseLevel, DietQuality


app = FastAPI()

df = pd.read_csv("health_generated.csv")
target = "risk"
model = train_naive_bayes(df, target)

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

    prediction = predict_naive_bayes(model, input_data)
    return {"prediction": prediction}
