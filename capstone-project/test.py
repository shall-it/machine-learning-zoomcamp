import requests

url = 'http://localhost:30080/predict'

test_urls = [
    "https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/main/capstone-project/blood-cell-images-for-cancer-detection-prepared/test/myeloblast/MYO_1212.jpg",
    "https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/main/capstone-project/blood-cell-images-for-cancer-detection-prepared/test/seg_neutrophil/NGS_1600.jpg",
    "https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/main/capstone-project/blood-cell-images-for-cancer-detection-prepared/test/basophil/BA_260100.jpg"
]


def get_risk_level(top_class):
    """Define risk level based on the predicted class"""
    risk_mapping = {
        'myeloblast': '🔴 HIGH RISK OF LEUKEMIA',
        'seg_neutrophil': '🟡 MIDDLE RISK OF LEUKEMIA', 
        'basophil': '🟢 LOW RISK OF LEUKEMIA',
        'erythroblast': '🟢 LOW RISK OF LEUKEMIA',
        'monocyte': '🟢 LOW RISK OF LEUKEMIA'
    }
    return risk_mapping.get(top_class, 'Unknown risk')


def send_request(image_url):
    try:
        request_data = {"url": image_url}
        response = requests.post(url, json=request_data, timeout=5)
        return response.json()
    except Exception as e:
        return f"Error: {e}"

print("🧪 Testing with the images from the different classes")
print("=" * 60)

for i, image_url in enumerate(test_urls, 1):
    print(f"\nTest #{i}")
    print(f"🔗 URL: {image_url}")
    
    result = send_request(image_url)
    
    if isinstance(result, dict) and 'top_class' in result:
        result['risk_level'] = get_risk_level(result['top_class'])
        print(f"Predicted class: {result['top_class']}")
        print(f"Probability: {result['top_probability']}")
        print(f"Risk level: {result['risk_level']}")
    else:
        print(f"Error: {result}")
    
    print("-" * 60)