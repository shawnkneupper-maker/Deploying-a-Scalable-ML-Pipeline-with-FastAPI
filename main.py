from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from ml.data import process_data
from ml.model import inference

# Load trained artifacts
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

# Categorical features used during training
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

app = FastAPI()

# Input schema
class CensusInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# ✅ Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello from the API!"}


# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: CensusInput):

    input_df = pd.DataFrame([data.dict()])

    # Rename columns to match training dataset
    input_df = input_df.rename(columns={
        "marital_status": "marital-status",
        "native_country": "native-country"
    })

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    pred = inference(model, X)

    result = lb.inverse_transform(pred)[0]

    return {"prediction": result}
