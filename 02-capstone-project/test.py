import requests

host = "price-prediction-serving-env-west-3.elasticbeanstalk.com"
url = f'http://{host}/predict'

ride_info = {
    "job": 3.1, 
    "surge_multiplier": 1.1, 
    "latitude": 42.3426,
    "longitude": -71.1006,
    "temperature": 34.5,
    "apparenttemperature": 28.8,
    "precipintensity": 0.01,
    "precipprobability": 0.01,
    "humidity": 0.85,
    "windspeed": 8.67,
    "windgust": 9.12,
    "visibility": 8,
    "source": "Back Bay",
    "name": "UberX",
    "hour": 10,
    "day": 26
    }

response = requests.post(url, json=ride_info).json()

print(response)