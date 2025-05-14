import requests
import json

# Test data
test_query = {
    "max_price": 300,
    "max_school_pov": 50,
    "min_facilities": 2,
    "comments": "looking for a quiet neighborhood"
}

# Send POST request to the API
response = requests.post(
    "http://localhost:8000/suggest",
    json=test_query
)

# Print results
print("Status code:", response.status_code)
print("\nRaw response text:")
print(response.text)
print("\nTrying to parse JSON response:")
try:
    json_response = response.json()
    print(json.dumps(json_response, indent=2))
except json.JSONDecodeError as e:
    print("Failed to parse JSON:", e) 