#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

patient_id_1 = 'abc-123'
patient1 = {
    "age": 45,
    "sex": 1,
    "cp": 3,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 2
}

patient_id_2 = 'def-456'
patient2 = {
    "age": 35,
    "sex": 0,
    "cp": 2,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 140,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 1,
    "thal": 3
}

def send_prediction(patient_data, patient_id):
    try:
        response = requests.post(url, json=patient_data)
        response.raise_for_status() 
        prediction = response.json()
        print(f"Response for patient {patient_id}: {prediction}")
        
        if prediction.get('hypertension', False):
            print(f'Sending an appointment email to {patient_id}')
        else:
            print(f'NOT sending an appointment email to {patient_id}')
    except requests.exceptions.RequestException as e:
        print(f"Request failed for patient {patient_id}: {e}")
    except ValueError:
        print(f"Invalid response format for patient {patient_id}")

send_prediction(patient1, patient_id_1)
send_prediction(patient2, patient_id_2)