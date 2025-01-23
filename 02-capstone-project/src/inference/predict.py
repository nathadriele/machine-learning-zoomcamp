import pickle
from flask import Flask, request, jsonify

model_file = 'price_prediction.bin'

try:
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{model_file}' not found. Ensure the file exists in the correct directory.")

app = Flask('uber_price_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the price of an Uber ride based on input data.
    """
    try:
        ride_info = request.get_json()

        if not isinstance(ride_info, dict):
            return jsonify({'error': 'Invalid input format. Expected a JSON object.'}), 400

        X = dv.transform([ride_info])

        y_pred = model.predict(X)

        result = {
            'predicted_price': round(float(y_pred[0]), 2)
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)