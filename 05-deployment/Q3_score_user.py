import pickle
 
def load_model(name: str):
    with open(name, "rb") as f_in:
        return pickle.load(f_in)

dv = load_dv(name="dv.bin")    
model = load_model("model1.bin")

user = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([user])
y_pred = model.predict_proba(X)[0, 1]
print(y_pred)