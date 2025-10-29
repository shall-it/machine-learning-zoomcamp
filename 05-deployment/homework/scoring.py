
import requests

url = 'http://localhost:9696/predict'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

score = requests.post(url, json=client).json()

print('Response:', score)

if score['score_probability'] >= 0.5:
    print('Converted')
else:
    print('Not converted')


# uv run python scoring.py