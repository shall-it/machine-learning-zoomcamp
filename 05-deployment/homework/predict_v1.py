import pickle


with open('pipeline_v1.bin', 'rb') as f_in:
  dv, model = pickle.load(f_in)


client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

def predict_single(client, dv, model):
  X = dv.transform([client])
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]


prediction = predict_single(client, dv, model)
churn = prediction >= 0.5

result = {
    'conversion_probability': float(prediction),
    'converted': bool(churn),
}

print(result)