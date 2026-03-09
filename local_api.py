import requests

url = "http://127.0.0.1:8000"

# Test GET
print("GET request")
response = requests.get(url)
print("Status Code:", response.status_code)
print("Result:", response.json())


# Correct input format
input_data = {
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

print("\nPOST request")

response = requests.post(
    f"{url}/predict",
    json=input_data
)

print("Status Code:", response.status_code)
print("Result:", response.json())
