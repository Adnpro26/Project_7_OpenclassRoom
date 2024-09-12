import streamlit as st

import pandas as pd
import numpy as np
import requests

import json

import pickle
import joblib

import os

import json

# Define the URL of the Flask API
API_URL = 'https://projectdeplapi-785f3fb8a900.herokuapp.com//predict'

# Titre de l'application
st.title("Analyse risque de crédit")

# Texte d'introduction
st.write("Bienvenue sur le simulateur du risque crédit")

# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))
#data_B_path = os.path.join(script_dir, 'Customers_Base.csv')
data_B_path = os.path.join(script_dir, 'data_test_features.csv')
#data_test_features.csv
customer_base = pd.read_csv(data_B_path) 

#print(len(customer_base))
#client_id = 307844
#customer_base = customer_base[customer_base["SK_ID_CURR"]==client_id]


customer_base = customer_base.set_index("SK_ID_CURR")
index_clients = customer_base.index

client_id = st.selectbox("ID du client", index_clients)

Customer = customer_base.loc[[client_id]]

Customer = Customer.to_json()

        
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
    #prediction_proba = response.json()['proba']
    print('Prediction:', prediction[0])

    predicted_class = prediction[0]
        
    #print('Prediction_proba:', prediction_proba)
else:
    print('Error:', response.text)


if st.button("Faire une prédiction"):
    st.subheader('Selon le modèle de prédiction : ')
    if predicted_class == 1:
        st.write(f"**le client est considéré comme risqué.**")
        
    else:
        st.write(f"**le client est considéré comme non risqué.**")
        
    