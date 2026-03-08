from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # or pickle

# Load your trained model and encoder
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")  # if you saved a label binarizer

app = FastAPI()

# Define input data model
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

# ✅ GET request on root
@app.get("/")
def read_root():
    return {"message": "Hello from the API!"}

# ✅ POST request for inference
@app.post("/predict")
def predict(input_data: CensusInput):
    # Convert input to DataFrame
    import pandas as pd
    input_df = pd.DataFrame([input_data.dict()])

    # Process data (use your process_data function)
    from ml.data import process_data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    from ml.model import inference
    pred = inference(model, X)

    # Convert prediction to readable label
    result = lb.inverse_transform(pred)[0] if lb else str(pred[0])
    return {"prediction": result}
