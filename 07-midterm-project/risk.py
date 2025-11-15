
import requests

url = 'http://localhost:9696/predict'

person = {
    "id": 5,
    "age": 26,
    "gender": "female",
    "region": "europe",
    "income_level": "lower-mid",
    "education_level": "bachelor",
    "daily_role": "full-time_employee",
    "device_hours_per_day": 13.07,
    "phone_unlocks": 199,
    "notifications_per_day": 91,
    "social_media_mins": 147,
    "study_mins": 60,
    "physical_activity_days": 1.0,
    "sleep_hours": 4.197962,
    "sleep_quality": 2.786098,
    "anxiety_score": 7.028125,
    "depression_score": 15.0,
    "stress_level": 9.448757,
    "happiness_score": 4.2,
    "focus_score": 70.0,
    "device_type": "android",
    "productivity_score": 65.299301,
    "digital_dependence_score": 48.4
}


response = requests.post(url, json=person)

risk = response.json()

print('Response:', risk)

if risk['risk'] >= 0.5:
    print('High risk! Time to take care of your digital and mental health')
else:
    print('Congratulations! Your digital and mental health are OK, risk is low')

# uv run python risk.py
