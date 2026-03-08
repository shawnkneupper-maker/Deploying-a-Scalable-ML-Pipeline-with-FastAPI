# ml/model.py
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score
import pandas as pd

# Train model
def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model with feature scaling.
    Returns the trained model and the scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000)  # increase iterations
    model.fit(X_scaled, y_train)

    # Return both the model and the scaler
    return model, scaler

# Run inference
def inference(model_scaler, X):
    """
    Run model predictions on scaled features.
    """
    model, scaler = model_scaler
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return preds

# Compute metrics
def compute_model_metrics(y, preds):
    """
    Compute precision, recall, fbeta with zero_division=1
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta

# Save model + scaler + encoders
def save_model(model_scaler, encoder, lb):
    model, scaler = model_scaler
    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(encoder, "model/encoder.pkl")
    joblib.dump(lb, "model/lb.pkl")

# Load model + scaler + encoders
def load_model():
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")
    return (model, scaler), encoder, lb
