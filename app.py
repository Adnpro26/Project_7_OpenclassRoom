from flask import Flask, request, jsonify
import pickle
import numpy as np

import pandas as pd
#import imblearn
import lightgbm as lgb
import joblib

import json


app = Flask(__name__)


#with open('/home/aoutanine/Project_7_OpenclassRoom/best_model_parameters.pkl', 'rb') as model_file:
    #model = pickle.load(model_file)

model = joblib.load('best_model_parameters.pkl')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
  
    # Preprocess data if needed
    # Example: Convert data to numpy array
    features = data["features"]

    df_json_text = pd.DataFrame([features])
    df_json_text = df_json_text.to_dict(orient='list')
    n = 0
    resultat = {}
    label_values = []
    values_customer = []
    for cle, texte in df_json_text.items():
        for elements in texte:
            element_to_dict = json.loads(elements)
            for c, t in element_to_dict.items():
                label_values.append(c)
                for c1,t1 in t.items():
                    values_customer.append(t1)

    df_customers = pd.DataFrame([values_customer], columns=label_values)

    #type_data = type(features)

    #print(features)
    
    # Make prediction
    #prediction = model.predict(features.reshape(1, -1))
    prediction = model.predict(df_customers)
    
    #Return prediction
    return jsonify({'prediction': prediction.tolist()})
    #return jsonify({'prediction': features})

    #return features

if __name__ == '__main__':
    app.run(debug=True)
