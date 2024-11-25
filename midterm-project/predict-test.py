#!/usr/bin/env python
# coding: utf-8

import requests
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("predict-test.log")
    ]
)

API_URL = 'http://localhost:9696/predict'

patients = [
    {
        "id": 'abc-123',
        "data": {
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
    },
    {
        "id": 'def-456',
        "data": {
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
    }
]

def send_prediction(patient_data, patient_id):
    """
    Sends a prediction request to the Flask API and handles the response.

    Parameters:
    - patient_data (dict): The feature data for the patient.
    - patient_id (str): The unique identifier for the patient.
    """
    try:
        logging.info(f"Sending prediction request for patient {patient_id}")
        response = requests.post(API_URL, json=patient_data)
        response.raise_for_status()

        prediction = response.json()
        logging.info(f"Received response for patient {patient_id}: {prediction}")

        if prediction.get('hypertension', False):
            logging.info(f"Sending an appointment email to patient {patient_id}")
        else:
            logging.info(f"NOT sending an appointment email to patient {patient_id}")

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred for patient {patient_id}: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred for patient {patient_id}: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred for patient {patient_id}: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"General request exception for patient {patient_id}: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decoding failed for patient {patient_id}: {json_err}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for patient {patient_id}: {e}")

def main():
    """
    Main function to iterate through patients and send prediction requests.
    """
    for patient in patients:
        patient_id = patient.get("id")
        patient_data = patient.get("data")
        
        if not patient_id or not patient_data:
            logging.warning(f"Missing ID or data for a patient entry: {patient}")
            continue
        
        send_prediction(patient_data, patient_id)

if __name__ == "__main__":
    main()
