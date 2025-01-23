import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'price_prediction.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('uber_price_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    ride_info = request.get_json()

    X = dv.transform([ride_info])
    y_pred = model.predict(X)

    result = {
        'price': round(float(y_pred),2),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)