import requests

url = 'http://localhost:30080/predict'

request_data = {
    "url": "https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/main/capstone-project/blood-cell-images-for-cancer-detection-prepared/test/myeloblast/MYO_0042.jpg"
}


def get_risk_level(top_class):
    """Define risk level based on the predicted class"""
    risk_mapping = {
        'myeloblast': '🔴 HIGH RISK OF CANCER',
        'seg_neutrophil': '🟡 MIDDLE RISK OF CANCER', 
        'basophil': '🟢 LOW RISK OF CANCER',
        'erythroblast': '🟢 LOW RISK OF CANCER',
        'monocyte': '🟢 LOW RISK OF CANCER'
    }
    return risk_mapping.get(top_class, 'Unknown risk')


def send_request(_):
    try:
        response = requests.post(url, json=request_data, timeout=5)
        return response.json()
    except Exception as e:
        return f"Error: {e}"


result = send_request(None)

result['risk_level'] = get_risk_level(result['top_class'])

print(f"Predicted class: {result['top_class']}")
print(f"Risk level: {result['risk_level']}")
