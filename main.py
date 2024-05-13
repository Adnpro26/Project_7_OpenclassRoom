import pandas as pd
import imblearn
import lightgbm as lgb
data = pd.read_csv("test_data.csv")
#print(data.head())

import joblib

model = joblib.load('best_model_parameters.pkl')


y_predict = model.predict(data)

print(y_predict)
