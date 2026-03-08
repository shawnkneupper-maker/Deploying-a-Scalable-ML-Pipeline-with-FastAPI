# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model
)

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

def main():
    # Load data
    data = pd.read_csv("data/census.csv")

    # Split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Process test data
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train model (returns model + scaler)
    model_scaler = train_model(X_train, y_train)

    # Run inference on test set
    preds = inference(model_scaler, X_test)

    # Compute overall metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {fbeta:.4f}")

    # Save model, scaler, and encoders
    save_model(model_scaler, encoder, lb)
    print("Model saved to model/model.pkl")
    print("Encoder saved to model/encoder.pkl")
    print("Scaler saved to model/scaler.pkl")

    # Compute slice metrics
    with open("slice_output.txt", "w") as f:
        for feature in cat_features:
            categories = test[feature].unique()
            for category in categories:
                slice_df = test[test[feature] == category]
                if slice_df.empty:
                    continue

                X_slice, y_slice, _, _ = process_data(
                    slice_df,
                    categorical_features=cat_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb
                )

                preds_slice = inference(model_scaler, X_slice)
                precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

                line1 = f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}\n"
                line2 = f"{feature}: {category}, Count: {len(slice_df)}\n"

                f.write(line1)
                f.write(line2)

    print("Slice metrics saved to slice_output.txt")

if __name__ == "__main__":
    main()
