import requests

BASE_URL = 'https://example-api-endpoint.amazonaws.com'
ENDPOINT = '/test/predict'

url = f'{BASE_URL}{ENDPOINT}'

data = {'url': 'http://example-image-url.com'}

def make_prediction(api_url, payload):
    """
    Sends a POST request to the given API URL with the specified payload.

    Args:
        api_url (str): The full API URL.
        payload (dict): The JSON data to send in the request.

    Returns:
        dict: The response JSON as a dictionary.
    """
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

result = make_prediction(url, data)

if result:
    print(result)
else:
    print("Failed to get a valid response from the API.")
