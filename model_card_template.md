# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Name: Census Income Classifier
Purpose: This model predicts whether an individual earns more than $50K per year based on census data.
Model Type: Logistic Regression classifier
Inputs: Features from census data including:
workclass
education
marital-status
occupation
relationship
race
sex
native-country
numerical catagories like age, hours-per-week, capital-gain, etc.
Output: Binary classification — >50K or <=50K.

## Intended Use
Use cases: Provides Estimate income categories for demographic analysis, research, or policy studies.
Target Audience: Data scientists, analysts, or researchers working with structured census-like data.
Not intended for: Hiring, Financial lending decisions, or other high-stakes decisions, as it may reflect societal biases.

## Training Data
Source: U.S. Census income dataset (adult dataset).
Size: Approximately 32,000 records for training, 8,000 for testing (your train/test split).
Preprocessing:
Categorical features one-hot encoded
Numerical features scaled
Target binarized (>50K = 1, <=50K = 0)

## Evaluation Data
Precision: 0.7339
Recall: 0.5652
F1-score: 0.6386


## Metrics
workclass: Private, Count: 4595 | Precision: 0.7381 | Recall: 0.6245 | F1: 0.6766
workclass: Self-emp-not-inc, Count: 495 | Precision: 0.7333 | Recall: 0.5203 | F1: 0.6087
education: Bachelors, Count: 3000 | Precision: 0.7500 | Recall: 0.6200 | F1: 0.6790
...

## Ethical Considerations
Predictions are estimates probabilty, not individual judgments.
This model should not be used for legal, financial, or employment decisions.
Consider fairness and bias mitigation if deploying in sensitive contexts.

## Caveats and Recommendations
 How to Use
```python
from ml.model import load_model, inference
from ml.data import process_data

# Load trained model and encoders
model, encoder, lb = load_model()

# Process new data
X_new, _, _, _ = process_data(new_data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

# Get predictions
predictions = inference(model, X_new)
