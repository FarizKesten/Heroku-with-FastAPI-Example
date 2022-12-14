import requests
import json

input = {
            'age': 40,
            'workclass': 'Private',
            'fnlgt': 140000,
            'education': 'Doctorate',
            'marital-status': 'Never-married',
            'occupation': 'Prof-specialty',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'hours-per-week': 60,
            'native-country': 'United-States'
        }

r = requests.post('https://deploy-ml-model-on-heroku.herokuapp.com/', json=input)

assert r.status_code == 200

print("Status code:", r.status_code)
print("input:", json.dumps(input, indent=2))
print("Prediction:",  r.json())