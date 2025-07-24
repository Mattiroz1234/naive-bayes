from fastapi import FastAPI
import pandas as pd
from .training import train_naive_bayes

app = FastAPI()

df = pd.read_csv("app/data/health_generated.csv")
model = train_naive_bayes(df, "risk")

@app.get("/model")
def get_model():
    return model
