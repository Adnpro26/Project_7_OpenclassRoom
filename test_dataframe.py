import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import requests

import json

import pickle
import joblib

import os

import shap

# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))

data_B_path = os.path.join(script_dir, 'data_test_features.csv')
#data_test_features.csv
customer_base = pd.read_csv(data_B_path) 
customer_base = customer_base.set_index("SK_ID_CURR")

# Construire le chemin absolu pour le fichier image
file_path = os.path.join(script_dir, 'shap_values.pkl')


with open(file_path, 'rb') as file:
    shap_values = pickle.load(file)


#explainer_values.pkl

file_explain = os.path.join(script_dir, 'explainer_values.pkl')


with open(file_path, 'rb') as file:
    file_explain = pickle.load(file)
# Construire le chemin absolu pour le fichier image
file_index = os.path.join(script_dir, 'test_data_index.pkl')


with open(file_index, 'rb') as file:
    index_values = pickle.load(file)

print(type(index_values))
client_id = 307844

print(type(client_id))

#Client_index = index_values.loc[client_id]

index_ligne = index_values[index_values == client_id].index

print(index_ligne)
print(type(file_explain))

tableau_filtre = file_explain[index_ligne]

print(tableau_filtre)

expected_value = -0.31125192035533533

#shap.initjs()
force_plot = shap.force_plot(expected_value, shap_values[index_ligne], customer_base.loc[[client_id]])

# Obtenir le chemin absolu du dossier contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin absolu pour le fichier image
html_path = os.path.join(script_dir, 'force_plot.html')
# Sauvegarder le graphique SHAP sous forme d'image HTML
shap.save_html(html_path, force_plot)

# Afficher le graphique interactif dans Streamlit
with open(html_path, "r") as f:
    html_code = f.read()

# Afficher le graphique HTML dans Streamlit (dynamique)
st.components.v1.html(html_code, height=400)