# API de Prédiction de Score de Crédit

## Objectif du Projet

Cette API est conçue pour fournir un score de crédit basé sur un modèle de machine learning. Elle utilise Flask pour exposer un service RESTful qui permet aux utilisateurs de soumettre des données et de recevoir des prédictions en temps réel. Ce projet est destiné à aider les institutions financières à évaluer le risque de crédit des clients potentiels de manière automatisée et efficace.

## Structure des Dossiers

├── app.py # Fichier principal de l'API Flask 
├── best_model_parametersl.pkl # Modèle de machine learning pré-entraîné 
├── requirements.txt # Liste des dépendances du projet 
├── Procfile # Fichier de configuration pour le déploiement sur Heroku 
├── README.md # Description du projet et instructions 
├── app.py # script permet d'envoyez une requête POST avec les données des caractéristiques au format JSON à l'endpoint 
├── request.py # utilisé pour répondre à l'API en envoyant une requête POST avec des données
├── Streamtest_Api.py # déploiement de l'api via streamlit
├── .gitignore #ndiquer les fichiers et dossiers que Git doit ignorer et ne pas suivre dans le contrôle de version