import pickle
import os
from flask import Flask, jsonify, request, abort

app = Flask("score_user")

def load_pickle_file(file_path: str):
    try:
        with open(file_path, "rb") as f_in:
            return pickle.load(f_in)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    except pickle.UnpicklingError:
        raise ValueError(f"Error: The file {file_path} could not be unpickled.")

def get_dv_model():
    dv = load_pickle_file("dv.bin")    
    model_name = os.getenv("MODEL_NAME")
    print(f"Using model: {model_name}")
    model = load_pickle_file(model_name)
    return dv, model

@app.route("/score", methods=["POST"])
def score_user():
    user = request.get_json()
    
    if not user or not isinstance(user, dict):
        abort(400, description="Invalid input: Expected a JSON object with user data.")

    try:
        dv, model = get_dv_model()
    except (FileNotFoundError, ValueError) as e:
        abort(500, description=f"Model loading error: {str(e)}")
    
    try:
        X = dv.transform([user])
        y_pred = model.predict_proba(X)[0, 1]
    except Exception as e:
        abort(500, description=f"Prediction error: {str(e)}")
    
    result = {
        "score": y_pred
    }
    
    return jsonify(result)
