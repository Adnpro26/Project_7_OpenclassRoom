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


print(type(shap_values))

print(type(customer_base))

fig, ax = plt.subplots()
shap.summary_plot(shap_values, customer_base, plot_type="bar")

# Exemple de beeswarm plot avec SHAP

#shap.plots.beeswarm(shap_values, show=False)  

# Utilisez Streamlit pour afficher le graphique
st.pyplot(fig)