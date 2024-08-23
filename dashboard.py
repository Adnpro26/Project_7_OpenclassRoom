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

from PIL import Image

# Titre de l'application
st.title("Analyse risque de crédit")

# Texte d'introduction
st.write("Bienvenue sur le simulateur du risque crédit")


# Méthode 3: Redimensionner l'image avec PIL avant d'afficher
image = Image.open('/home/aoutanine/Project_7_OpenclassRoom/logo_image.jpg')
image = image.resize((100, 60))  # Ajuster la taille souhaitée
st.sidebar.image(image)
st.sidebar.title("Credit Simulator")
st.sidebar.header("Paramètres")
Id_Value = st.sidebar.text_input("Entrez votre identifiant")
st.sidebar.selectbox("Sélectionner type export", ["PDF", "HTML", "e-MAIL"])



# Afficher les informations saisies
st.write(f"Identifiant : {Id_Value}")


# Fonction pour afficher une jauge colorée
def taux_endettement(endettement):
    # Créer l'élément de jauge
    indicator = go.Indicator(
        mode="gauge+number",
        value=endettement,
        delta={'reference': 100/3},
        gauge={
            'axis': {'visible': True, 'range': [None, 100]},
            'steps': [
                {'range': [0, 100], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 1,
                'value': 100
            }
        },
        title={'text': "Taux d'endettement"},
        domain={'x': [0, 0.5], 'y': [0, 1]}
    )
    
    # Encapsuler l'indicateur dans une figure
    fig = go.Figure(indicator)
    st.plotly_chart(fig)


# Fonction pour afficher une jauge colorée
def afficher_jauge(score):
    # Créer l'élément de jauge
    indicator = go.Indicator(
        mode="gauge+number",
        value=score,
        delta={'reference': 100},
        gauge={
            'axis': {'visible': True, 'range': [None, 100]},
            'steps': [
                {'range': [0, 100], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 1,
                'value': 100
            }
        },
        title={'text': "Score de crédit"},
        domain={'x': [0, 0.5], 'y': [0, 1]}
    )
    
    # Encapsuler l'indicateur dans une figure
    fig = go.Figure(indicator)
    st.plotly_chart(fig)


# Fonction pour afficher l'importance des features
def afficher_importance_features(feature_importance_locale, feature_importance_globale):
    st.subheader("Importance des features (locale)")
    st.bar_chart(feature_importance_locale)
    
    st.subheader("Importance des features (globale)")
    st.bar_chart(feature_importance_globale)


# Charger le fichier CSV
customer_info = pd.read_csv("/home/aoutanine/Project_7_OpenclassRoom/data_test_Costumers.csv") 
customer_info = customer_info.set_index("SK_ID_CURR")

customer_info = customer_info.drop(columns=['Replacement_Occupation'])



customer_base = pd.read_csv("/home/aoutanine/Project_7_OpenclassRoom/Customers_Base.csv") 
customer_base = customer_base.set_index("SK_ID_CURR")
index_clients = customer_base.index


customer_info = customer_info.loc[customer_info.index.isin(index_clients)]

client_id = st.selectbox("ID du client", index_clients)

Customer = customer_base.loc[[client_id]]

Customer_details = customer_info.loc[[client_id]]

contract_type = Customer_details['NAME_CONTRACT_TYPE'].values[0]

filtered_contracts= customer_info[customer_info['NAME_CONTRACT_TYPE'] == contract_type]
scope_clients = filtered_contracts.index.tolist()


Customer_details_transposed = Customer_details.T
Customer_details_transposed.columns = ['Valeur'] 

def adjust_column_widths(df, width=100):
    return df.style.set_properties(**{'width': f'{width}px'})


def highlight_max(s):
    return ['background-color: yellow' if v == s.max() else '' for v in s]

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

gender_code = Customer_details['CODE_GENDER'].values[0]
age_value = Customer['age'].values[0]
age_rounded = round(age_value)

tx_endettement = Customer_details['Taux endettement'].values[0]
tx_endettement = tx_endettement * 100

revenu_mesnuel = Customer_details['Revenue mensuel'].values[0]
revenu_rounded = round(revenu_mesnuel)

type_revenu = Customer_details['OCCUPATION_TYPE'].values[0]

statut_famille = Customer_details['NAME_FAMILY_STATUS'].values[0]
type_education = Customer_details['NAME_EDUCATION_TYPE'].values[0]
type_propriete = Customer_details['NAME_HOUSING_TYPE'].values[0]
#NAME_HOUSING_TYPE

score = predicted_proba[0][class_pred]
score = score*100

col1, col2 = st.columns(2) 
with col1:
    st.subheader('Genre : ' + str(gender_code))

with col2:
    st.subheader(f'Âge : {age_rounded}')  


col1, col2 = st.columns(2) 
with col1:
    st.markdown(f"**Type revenu :** {type_revenu}")

with col2:
    st.markdown(f"**Revenu mensuel :** ${revenu_rounded:,.2f}")

col1, col2= st.columns(2) 
with col1:
    #st.text('Statut : ' + str(statut_famille))
    st.markdown(f"**Statut :** {statut_famille}")

#with col2:
    #st.text('Niveau étude : ' + str(type_education)) 

with col2:
    st.markdown(f"**Type de logement :** {type_propriete}") 



st.markdown(f"**Niveau étude :** {type_education}")



age_stats = customer_base['age'].describe()
revenue_stats = customer_info['Revenue mensuel'].describe()

st.subheader('Statistiques Descriptives')

col1, col2= st.columns(2)
with col1:
    st.write("### Statistiques sur l'âge")
    st.write(age_stats)
with col2:
    st.write("### Statistiques sur le revenu")
    st.write(revenue_stats)


variables_modifiables = ['ext_source_2', 'amt_credit', 'amt_goods_price', 'duree_moyenne_credit']


with st.form(key='update_form'):
    # Champs du formulaire
    revenu_st = st.number_input("Revenue mensuel", value=Customer_details['Revenue mensuel'].values[0])
    credit_st = st.number_input("Mensualité crédit", value=Customer_details['Mensualité crédit'].values[0])#Mensualité crédit
    autres_charges_st = st.number_input("Autres charges mensuelles", 0)#Taux endettement
    # Sélectionner les variables à modifier avec multiselect
    choix_variables = st.multiselect("Sélectionnez les variables à modifier", variables_modifiables)

    if choix_variables:
        if 'ext_source_2' in choix_variables:
            ext_source_2_st = st.number_input("Entrez la nouvelle valeur de ext_source_2", value=Customer['ext_source_2'].values[0])
        if 'amt_credit' in choix_variables:
            amt_credit_st = st.number_input("Entrez la nouvelle valeur de amt_credit", value=Customer['amt_credit'].values[0])
        if 'amt_goods_price' in choix_variables:
            amt_goods_price_st = st.number_input("Entrez la nouvelle valeur de amt_goods_price", value=Customer['amt_goods_price'].values[0])
        if 'duree_moyenne_credit' in choix_variables:
            duree_credit_st = st.number_input("Entrez la nouvelle valeur de duree_moyenne_credit", value=Customer['duree_moyenne_credit'].values[0])
            
            
    add_button = st.form_submit_button(label='Ajouter nouvelles valeurs')
        
    
    # Sélectionner la variable à modifier
    #choix_variable = st.selectbox("Choisissez la variable à modifier", variables_modifiables)

    # Bouton de soumission
    submit_button = st.form_submit_button(label='Enregistrer')


    if submit_button:
        # Vérifier si la variable 'ext_source_2' est dans les variables sélectionnées par l'utilisateur
        if 'ext_source_2' in choix_variables:
            if ext_source_2_st != Customer['ext_source_2'].values[0]:
                Customer['ext_source_2'] = ext_source_2_st
        if 'amt_credit' in choix_variables:
            if amt_credit_st != Customer['amt_credit'].values[0]:
                Customer['amt_credit'] = amt_credit_st
        if 'amt_goods_price' in choix_variables:
            if amt_goods_price_st != Customer['amt_goods_price'].values[0]:
                Customer['ext_source_2'] = amt_goods_price_st
        if 'duree_moyenne_credit' in choix_variables:
            if ext_source_2_st != Customer['duree_moyenne_credit'].values[0]:
                Customer['duree_moyenne_credit'] = duree_credit_st
        
        ext_src_thrd_st = (credit_st + autres_charges_st) / revenu_st
        tx_endettement_st = credit_st / revenu_st
        # Mettre à jour les données
        customer_info.loc[[client_id], ['Revenue mensuel', 'Mensualité crédit', 'Taux endettement']] = [revenu_st, credit_st, tx_endettement_st]
        
        # Sauvegarder les modifications
        #save_data(customer_info)
        
        # Calcul du taux en pourcentage
        Customer['ext_source_3'] = ext_src_thrd_st
        tx_endettement = tx_endettement_st * 100  # Multiplier par 100 pour obtenir un pourcentage
        # Affichage formaté avec le symbole %
        #st.subheader("Taux d'endettement réel : " + str(round(taux, 2)) + "%")

        # Faire des prédictions avec le modèle chargé
        Customer_np = Customer.values.reshape(1, -1)  # Reshape to (1, 120) for prediction
        predicted_class = loaded_model.predict(Customer_np)
        predicted_proba = loaded_model.predict_proba(Customer_np)
        score = predicted_proba[0][class_pred]
        score = score*100
        
        st.success("Données du client mises à jour avec succès !")
        


st.subheader('Classe Prédite')
if predicted_class == 1:
    st.write(f"**le client est considéré comme risqué.**")
else:
    st.write(f"**le client est considéré comme non risqué.**")

col1, col2 = st.columns(2) 
with col1:
    afficher_jauge(score)

with col2:    
    taux_endettement(tx_endettement)

bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
#customer_base_df = customer_base.copy()
customer_base_df = customer_base.loc[customer_base.index.isin(scope_clients)]
customer_base_df['AgeGroup'] = pd.cut(customer_base_df['age'], bins=bins, labels=labels, right=False)


with st.expander("Données du client"):
    colonnes_disponibles = customer_base.columns.tolist()
    feature_selectionnee = st.selectbox("Sélectionner une caractéristique (feature) à afficher", colonnes_disponibles)
    valeur_client = Customer[feature_selectionnee].values[0]
    
    st.subheader(f"Distribution de la caractéristique '{feature_selectionnee}'")
    fig, ax = plt.subplots()
    sns.histplot(customer_base[feature_selectionnee], kde=True, ax=ax)
    ax.axvline(valeur_client, color='r', linestyle='--', label=f'Client {client_id}')
    ax.legend()
    
    st.pyplot(fig)
    
    # Afficher les données du client
    st.subheader("Données du client")
    st.write(customer_base.loc[[client_id]])
    feature_bi_variee_x = st.selectbox("Sélectionner la première caractéristique pour l'analyse bivariée", colonnes_disponibles)
    feature_bi_variee_y = st.selectbox("Sélectionner la seconde caractéristique pour l'analyse bivariée", colonnes_disponibles)
    
    st.subheader(f"Analyse bivariée entre '{feature_bi_variee_x}' et '{feature_bi_variee_y}'")

    fig, ax = plt.subplots()
    sns.scatterplot(data=customer_base, x=feature_bi_variee_x, y=feature_bi_variee_y, ax=ax)
    ax.axvline(Customer[feature_bi_variee_x].values[0], color='r', linestyle='--')
    ax.axhline(Customer[feature_bi_variee_y].values[0], color='r', linestyle='--')
    ax.set_xlabel(feature_bi_variee_x)
    ax.set_ylabel(feature_bi_variee_y)
    ax.set_title(f"Scatter plot de {feature_bi_variee_x} vs {feature_bi_variee_y}")
    
    st.pyplot(fig)

    # Créer un boxplot des revenus en fonction des tranches d'âge
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='AgeGroup', y='amt_annuity', data=customer_base_df)
    # Ajouter le point pour le client sélectionné
    Customer_id_df = customer_base_df.loc[[client_id]]
    if not Customer.empty:
        age_group = Customer_id_df['AgeGroup'].values[0]
        annuite = Customer_id_df['amt_annuity'].values[0]
        plt.scatter(age_group, annuite, color='red', s=100, label=f'Client {client_id}', edgecolor='black')
    
    plt.annotate(f'Client {client_id}', 
                 xy=(age_group, annuite), 
                 xytext=(age_group, annuite + 5000), # Adjust vertical position for readability
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=10,
                 color='red')
    plt.title('Boxplot des annuitées par Tranche d\'Âge')
    plt.xlabel('Tranche d\'Âge')
    plt.ylabel('Annuitées')
    plt.legend()
    # Afficher le graphique avec Streamlit
    st.subheader(f"Analyse contrats de type : '{contract_type}'")
    st.pyplot(plt)


with st.form(key='my_form'):
    # Ajouter une sélection de choix
    option = st.radio(
        "Sélectionnez l'option",
        ("Demande de crédit accordé", "Demande de crédit non accordé")
    )
    
    # Ajouter un bouton de soumission
    submit_button = st.form_submit_button(label='Soumettre')
    
    # Afficher le résultat lorsque le bouton est cliqué
    if submit_button:
        if option == "Demande de crédit accordé":
            st.write("Avis favorable")
        else:
            st.write("Avis défavorable")