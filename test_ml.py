import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Example categorical features
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

# Load a small sample of data for testing
sample_data = pd.DataFrame({
    "age": [25, 38, 28],
    "workclass": ["Private", "Self-emp-not-inc", "Private"],
    "education": ["Bachelors", "HS-grad", "Bachelors"],
    "marital-status": ["Never-married", "Married-civ-spouse", "Never-married"],
    "occupation": ["Tech-support", "Exec-managerial", "Sales"],
    "relationship": ["Not-in-family", "Husband", "Not-in-family"],
    "race": ["White", "White", "Black"],
    "sex": ["Male", "Male", "Female"],
    "capital-gain": [0, 0, 0],
    "capital-loss": [0, 0, 0],
    "hours-per-week": [40, 50, 40],
    "native-country": ["United-States", "United-States", "United-States"],
    "salary": [0, 1, 0]
})


def test_process_data():
    """Test that process_data returns expected shapes and types."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert X.shape[0] == 3, "X should have 3 rows"
    assert y.shape[0] == 3, "y should have 3 rows"
    assert hasattr(encoder, "transform"), "Encoder should have a transform method"


def test_train_model_returns_logistic_regression():
    """Test that train_model returns a LogisticRegression instance."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model_tuple = train_model(X, y)
    model = model_tuple[0]  # get the LogisticRegression object
    assert isinstance(model, LogisticRegression), "Model should be LogisticRegression"


def test_compute_model_metrics_returns_values():
    """Test that compute_model_metrics returns 3 float values."""
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
