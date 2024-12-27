import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = "pipeline.joblib"
pipeline = joblib.load(model_path)

@app.route("/")
def home():
    return "Ford Car Price Prediction API - Welcome!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    {
        "model": "Focus",
        "year": 2019,
        "transmission": "Manual",
        "mileage": 8131,
        "fuelType": "Petrol",
        "tax": 145,
        "mpg": 58.9,
        "engineSize": 1.0
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    df_input = pd.DataFrame([data])

    try:
        pred = pipeline.predict(df_input)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": float(pred)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)