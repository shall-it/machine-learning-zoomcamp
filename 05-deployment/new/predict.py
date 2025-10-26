import pickle

with open ('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

customer = {
    'gender': 'male',
     'seniorcitizen': 0,
     'partner': 'no',
     'dependents': 'yes',
     'phoneservice': 'no',
     'multiplelines': 'no_phone_service',
     'internetservice': 'dsl',
     'onlinesecurity': 'no',
     'onlinebackup': 'yes',
     'deviceprotection': 'no',
     'techsupport': 'no',
     'streamingtv': 'no',
     'streamingmovies': 'no',
     'contract': 'month-to-month',
     'paperlessbilling': 'yes',
     'paymentmethod': 'electronic_check',
     'tenure': 6,
     'monthlycharges': 29.85,
     'totalcharges': 129.85
}

churn = pipeline.predict_proba(customer)[0, 1]

print('Prob of churning =', churn)

if churn >= 0.5:
    print('Send promo email')
else:
    print('Do not do anything')
