import pickle

def load_pickle_file(file_path: str):
    """Loads a pickle file from the given path."""
    try:
        with open(file_path, "rb") as f_in:
            return pickle.load(f_in)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    except pickle.UnpicklingError:
        raise ValueError(f"Error: The file {file_path} could not be unpickled.")

# Load the dictionary vectorizer and model
dv = load_pickle_file("dv.bin")    
model = load_pickle_file("model1.bin")

user = {"job": "management", "duration": 400, "poutcome": "success"}

# Transform the user data and make a prediction
X = dv.transform([user])
y_pred = model.predict_proba(X)[0, 1]
print(f"Predicted probability: {y_pred}")