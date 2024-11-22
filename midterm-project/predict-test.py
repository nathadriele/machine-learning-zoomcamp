#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

patient_id = 'abc-123'
patient1 = {
    "NumMedicalVisits": 5,
    "Cholesterol": 240,
    "BloodPressure": 130,
    "PhysicalActivity": 90,
    "SodiumIntake": 3500,
    "BMI": 28.4,
    "HypertensionPedigreeFunction": 1.2,
    "Age": 45
}

patient2 = {
    "NumMedicalVisits": 3,
    "Cholesterol": 200,
    "BloodPressure": 120,
    "PhysicalActivity": 150,
    "SodiumIntake": 2800,
    "BMI": 25.1,
    "HypertensionPedigreeFunction": 0.8,
    "Age": 35
}

response1 = requests.post(url, json=patient1).json()
print(f"Response for patient {patient_id}: {response1}")

if response1['hypertension']:
    print(f'Sending an appointment email to {patient_id}')
else:
    print(f'NOT sending an appointment email to {patient_id}')

response2 = requests.post(url, json=patient2).json()
print(f"Response for patient 2: {response2}")

if response2['hypertension']:
    print('Sending an appointment email to patient 2')
else:
    print('NOT sending an appointment email to patient 2')