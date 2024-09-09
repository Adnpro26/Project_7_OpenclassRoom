import streamlit as st

import pandas as pd
import numpy as np
import requests

import json

import pickle
import joblib


# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin absolu pour le fichier image
data_A_path = os.path.join(script_dir, 'data_test_Costumers.csv')
data_B_path = os.path.join(script_dir, 'Customers_Base.csv')

# Titre de l'application
st.title("Analyse risque de crédit")

# Texte d'introduction
st.write("Bienvenue sur le simulateur du risque crédit")




customer_base = pd.read_csv(data_B_path) 
customer_base = customer_base.set_index("SK_ID_CURR")
index_clients = customer_base.index

client_id = st.selectbox("ID du client", index_clients)

Customer = customer_base.loc[[client_id]]

#Customer = Customer.to_frame()

Customer_np = Customer.values.reshape(1, -1)  # Reshape to (1, 120) for prediction


loaded_model = joblib.load('best_model_parameters.pkl')

#print(type(loaded_model))

# Faire des prédictions avec le modèle chargé
predicted_class = loaded_model.predict(Customer_np)
predicted_proba = loaded_model.predict_proba(Customer_np)

# Afficher les résultats
print(f"Prédiction pour le client : {predicted_class[0]}")
class_pred = predicted_class[0]
print(f"Probabilités pour le client : {predicted_proba[0][class_pred]}")

#CODE_GENDER
#NAME_INCOME_TYPE

Customer_np = Customer.values.reshape(1, -1)  # Reshape to (1, 120) for prediction
predicted_class = loaded_model.predict(Customer_np)
predicted_proba = loaded_model.predict_proba(Customer_np)
score = predicted_proba[0][class_pred]
score = score*100
        


st.subheader('Classe Prédite')
if predicted_class == 1:
    st.write(f"**le client est considéré comme risqué.**")
else:
    st.write(f"**le client est considéré comme non risqué.**")