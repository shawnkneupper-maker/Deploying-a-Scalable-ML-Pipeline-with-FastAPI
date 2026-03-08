import requests

# URL of the running API
url = "http://127.0.0.1:8000"

# ✅ GET request
response = requests.get(url + "/")
print("GET request")
print("Status Code:", response.status_code)
print("Result:", response.json())

# ✅ POST request
sample_input = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post(url + "/predict", json=sample_input)
print("\nPOST request")
print("Status Code:", response.status_code)
print("Result:", response.json())
