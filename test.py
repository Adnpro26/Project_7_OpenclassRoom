
import pickle
import numpy as np

import pandas as pd
#import imblearn
import lightgbm as lgb
import joblib


#with open('/home/aoutanine/Project_7_OpenclassRoom/best_model_parameters.pkl', 'rb') as model_file:
    #model = pickle.load(model_file)

model = joblib.load('/home/aoutanine/Project_7_OpenclassRoom/best_model_parameters.pkl')

X_test=pd.read_csv("/home/aoutanine/Project_7_OpenclassRoom/test_data.csv") 

Customer = X_test.iloc[[4]]
print(type(Customer))

#Customer = Customer.values.tolist()

# Sample data for prediction
data = {
    'features': Customer
}

#print(X_test.index)
print(type(data["features"]))


y_pred = model.predict(data["features"])

print(y_pred)