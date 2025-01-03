import pickle
import os
from flask import Flask, jsonify, request, abort

app = Flask("score_user")

def load_pickle_file(file_path: str):
    """Loads a pickle file from the given path."""
    try:
        with open(file_path, "rb") as f_in:
            return pickle.load(f_in)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    except pickle.UnpicklingError:
        raise ValueError(f"Error: The file {file_path} could not be unpickled.")

def get_dv_model():
    """Loads the dictionary vectorizer and model."""
    dv = load_pickle_file("dv.bin")    
    model_name = os.getenv("MODEL_NAME", "model1.bin")
    print(f"Using model: {model_name}")
    model = load_pickle_file(model_name)
    return dv, model

@app.route("/score", methods=["POST"])
def score_user():
    """Endpoint to score a user based on the provided JSON payload."""
    user = request.get_json()
    
    # Validate input
    if not user or not isinstance(user, dict):
        abort(400, description="Invalid input: Expected a JSON object with user data.")

    # Load vectorizer and model
    try:
        dv, model = get_dv_model()
    except (FileNotFoundError, ValueError) as e:
        abort(500, description=f"Model loading error: {str(e)}")
    
    # Transform the input and make a prediction
    try:
        X = dv.transform([user])
        y_pred = model.predict_proba(X)[0, 1]
    except Exception as e:
        abort(500, description=f"Prediction error: {str(e)}")
    
    result = {
        "score": y_pred
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
