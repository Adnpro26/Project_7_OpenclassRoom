from flask import Flask, request, jsonify

import pandas as pd
#import imblearn
import lightgbm as lgb

#import pickle
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('/home/aoutanine/Project_7_OpenclassRoom/best_model_parameters.pkl')

def chargement_donnees():
    X_test=pd.read_csv("test_data.csv")
    donnees = X_test.to_dict(orient='records')
    return donnees

@app.route("/")
def home():
    return "Prédiction risque de crédit"

@app.route('/donnees-test', methods=['GET'])
def input_test_data():
    donnees = chargement_donnees()
    return jsonify(donnees)

@app.route('/predict', methods=['POST'])
def predict():
    
    input_data = request.json["donnnes"]

    
    prediction = model.predict(input_data)

    
    return jsonify({'prediction': prediction.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)