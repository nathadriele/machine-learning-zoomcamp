import pickle
import logging
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)

model_file = 'model.bin'

try:
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    logging.info('Model and vectorizer loaded successfully.')
except FileNotFoundError:
    logging.error(f'Model file {model_file} not found. Please ensure the file is available.')
    raise
except Exception as e:
    logging.error(f'Error loading model: {e}')
    raise

app = Flask('hypertension_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient = request.get_json()
        required_fields = [
            "NumMedicalVisits", "Cholesterol", "BloodPressure",
            "PhysicalActivity", "SodiumIntake", "BMI",
            "HypertensionPedigreeFunction", "Age"
        ]

        missing_fields = [field for field in required_fields if field not in patient]
        if missing_fields:
            return jsonify({"error": f"Missing fields in input: {', '.join(missing_fields)}"}), 400

        X = dv.transform([patient])
        y_pred = model.predict_proba(X)[0, 1]
        hypertension = y_pred >= 0.5

        result = {
            'hypertension_probability': float(y_pred),
            'hypertension': bool(hypertension)
        }

        return jsonify(result)
    except Exception as e:
        logging.error(f'Error during prediction: {e}')
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
