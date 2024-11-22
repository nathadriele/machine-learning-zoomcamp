#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

patient_id = 'abc-123'
patient = {
    "NumMedicalVisits": 5,
    "Cholesterol": 240,
    "BloodPressure": 130,
    "PhysicalActivity": 90,
    "SodiumIntake": 3500,
    "BMI": 28.4,
    "HypertensionPedigreeFunction": 1.2,
    "Age": 45
}

response = requests.post(url, json=patient).json()
print(response)

if response['hypertension']:
    print(f'Sending an appointment email to {patient_id}')
else:
    print(f'NOT sending an appointment email to {patient_id}')