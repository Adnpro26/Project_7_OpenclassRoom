import requests

import pandas as pd
import json

# Define the URL of the Flask API
API_URL = 'https://projectdeplapi-785f3fb8a900.herokuapp.com//predict'

X_test=pd.read_csv("/home/aoutanine/Project_7_OpenclassRoom/test_data.csv") 
Customer = X_test.iloc[[4]]
Customer = Customer.to_json()
#Customer = Customer.tolist()
# Sample data for prediction
data = {
    'features': Customer
}

#print(data)

# Send a POST request to the Flask API
response = requests.post(API_URL, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Get the prediction from the response
    prediction = response.json()['prediction']
    print('Prediction:', prediction)
else:
    print('Error:', response.text)
