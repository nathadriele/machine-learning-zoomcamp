import pandas as pd
import numpy as np
import joblib

pipeline = joblib.load("pipeline.joblib")

X1 = {
    "model": "Fiesta",
    "year": 2019,
    "transmission": "Manual",
    "mileage": 3265,
    "fueltype": "Petrol",
    "tax": 145,
    "mpg": 58.9,
    "enginesize": 1
}

X2 = {
    "model": "Focus",
    "year": 2018,
    "transmission": "Manual",
    "mileage": 9083,
    "fueltype": "Petrol",
    "tax": 150,
    "mpg": 57.7,
    "enginesize": 1
}

df_test = pd.DataFrame([X1, X2])
predictions = pipeline.predict(df_test)
print("X1 Predicted Price:", predictions[0])
print("X2 Predicted Price:", predictions[1])