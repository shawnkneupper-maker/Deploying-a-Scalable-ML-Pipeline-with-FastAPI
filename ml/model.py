import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : machine learning model
        Trained model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)
    return preds


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1 score.

    Inputs
    ------
    y : np.array
        Known labels.
    preds : np.array
        Predicted labels.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1)

    return precision, recall, fbeta


def save_model(model, encoder, lb,
               model_path="model/model.pkl",
               encoder_path="model/encoder.pkl",
               lb_path="model/lb.pkl"):
    """
    Save model and encoders using pickle.
    """

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    with open(lb_path, "wb") as f:
        pickle.dump(lb, f)


def load_model(model_path="model/model.pkl",
               encoder_path="model/encoder.pkl",
               lb_path="model/lb.pkl"):
    """
    Load model and encoders.
    """

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    with open(lb_path, "rb") as f:
        lb = pickle.load(f)

    return model, encoder, lb


def performance_on_categorical_slice(
        model, X, y, dataframe, categorical_feature):
    """
    Computes model performance on slices of the data defined by a categorical feature.

    Inputs
    ------
    model : trained model
    X : np.array
        Processed features
    y : np.array
        True labels
    dataframe : pd.DataFrame
        Original dataframe
    categorical_feature : str
        Feature to slice on
    """

    results = {}

    categories = dataframe[categorical_feature].unique()

    for category in categories:

        slice_df = dataframe[dataframe[categorical_feature] == category]

        X_slice = X[slice_df.index]
        y_slice = y[slice_df.index]

        preds = inference(model, X_slice)

        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        results[category] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        }

    return results
