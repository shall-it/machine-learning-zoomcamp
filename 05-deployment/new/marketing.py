churn = pipeline.predict_proba(customer)[0, 1]

print('Prob of churning =', churn)

if churn >= 0.5:
    print('Send promo email')
else:
    print('Do not do anything')