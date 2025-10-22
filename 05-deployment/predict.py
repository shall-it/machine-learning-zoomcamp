
# import

import pickle
from flask import Flask
from flask import request
from flask import jsonify


# parameters

model_file = 'model_C1.0.bin'


# load the model

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('churn')

@app.route('/predict', methods = ['GET'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    churn = y_pred >= 0.5

    result = {
        'churn_probability': y_pred,
        'churn': churn
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)