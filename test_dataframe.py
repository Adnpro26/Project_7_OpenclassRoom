import streamlit as st

import pandas as pd
import numpy as np
import requests

import json
import plotly.graph_objects as go
import shap

import pickle
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Analyse risque de crédit")

# Texte d'introduction
st.write("Bienvenue sur le simulateur du risque crédit")

# Créer des éléments interactifs
Id_Value = st.text_input("Entrez votre identifiant")


# Afficher les informations saisies
st.write(f"Identifiant : {Id_Value}")



# Fonction pour afficher une jauge colorée
def afficher_jauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Score de crédit"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))
    st.plotly_chart(fig)

# Fonction pour afficher l'importance des features
def afficher_importance_features(feature_importance_locale, feature_importance_globale):
    st.subheader("Importance des features (locale)")
    st.bar_chart(feature_importance_locale)
    
    st.subheader("Importance des features (globale)")
    st.bar_chart(feature_importance_globale)


# Charger le fichier CSV
customer_base = pd.read_csv("/home/aoutanine/Project_7_OpenclassRoom/Customers_Base.csv") 
customer_base = customer_base.set_index("SK_ID_CURR")
index_clients = customer_base.index

client_id = st.selectbox("ID du client", index_clients)
#client_id = int(client_id)
print(type(client_id))


#client_id = 333371

print(type(client_id))

Customer = customer_base.loc[client_id]
Customer = Customer.to_frame()

print(type(Customer))
Customer = Customer.values.reshape(1, -1)  # Reshape to (1, 120) for prediction

loaded_model = joblib.load('best_model_parameters.pkl')

#print(type(loaded_model))

# Faire des prédictions avec le modèle chargé
predicted_class = loaded_model.predict(Customer)
predicted_proba = loaded_model.predict_proba(Customer)

# Afficher les résultats
print(f"Prédiction pour le client : {predicted_class[0]}")
#class_pred = predicted_class[0]
print(f"Probabilités pour le client : {predicted_proba[0][predicted_class]}")

