import streamlit as st
import plotly.graph_objects as go

# Fonction pour afficher la jauge avec Plotly
def afficher_jauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={'axis': {'range': [0, 100]}},
        title={'text': "Score de risque"}
    ))
    st.plotly_chart(fig)

# Initialisation des variables simulées avec valeurs par défaut
predicted_class = 0
score = 75  # score de risque initial
tx_endettement = 35  # taux d'endettement initial

# Widgets pour permettre à l'utilisateur de saisir de nouvelles données
age = st.number_input("Entrez l'âge du client", min_value=18, max_value=100, value=35)
revenu = st.number_input("Entrez le revenu du client", min_value=0, max_value=100000, value=3000)
tx_endettement = st.number_input("Entrez le taux d'endettement (%)", min_value=0, max_value=100, value=tx_endettement)

# Bouton de soumission
submit_button = st.button("Soumettre les modifications")

# Simulation de mise à jour des données après modification par l'utilisateur
if submit_button:
    # Logique pour recalculer la classe prédite et le score en fonction du taux d'endettement
    if tx_endettement > 40:  # Critère arbitraire pour la prédiction
        predicted_class = 1
        score = 90  # Exemple : plus le taux d'endettement est élevé, plus le risque est élevé
    else:
        predicted_class = 0
        score = 60  # Exemple : moins le taux d'endettement est élevé, moins le risque est élevé

# Affichage de la classe prédite mise à jour
st.subheader('Classe Prédite')
if predicted_class == 1:
    st.write(f"**Le client est considéré comme risqué.**")
else:
    st.write(f"**Le client est considéré comme non risqué.**")

# Affichage de la jauge (une seule fois)
afficher_jauge(score)
