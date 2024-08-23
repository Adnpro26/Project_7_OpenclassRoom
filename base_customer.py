import pandas as pd
import numpy as np

import pickle
import joblib


# Charger le fichier CSV


test_data = pd.read_csv("/home/aoutanine/Project_7_OpenclassRoom/data_test_features.csv") 
liste_columns = test_data.columns


# Nombre d'échantillons aléatoires à extraire
n = 5000  # Par exemple, 5 clients aléatoires

# Extraire des échantillons aléatoires
customer_base = test_data.sample(n=n, random_state=1)

customer_base = customer_base.set_index("SK_ID_CURR")

# Afficher les échantillons aléatoires
customer_base.to_csv("/home/aoutanine/Project_7_OpenclassRoom/Customers_Base.csv", index=True)

client_id = 333371

print(type(client_id))

Customer = customer_base.loc[client_id]
Customer = Customer.to_frame()
print(type(customer_base))

loaded_model = joblib.load('best_model_parameters.pkl')

#print(type(loaded_model))

# Faire des prédictions avec le modèle chargé
predicted_class = loaded_model.predict(Customer)
predicted_proba = loaded_model.predict_proba(Customer)

# Afficher les résultats
print(f"Prédiction pour le client : {predicted_class[0]}")
#class_pred = predicted_class[0]
print(f"Probabilités pour le client : {predicted_proba[0][predicted_class]}")
