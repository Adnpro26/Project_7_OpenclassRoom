from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    
    # Preprocess data if needed
    # Example: Convert data to numpy array
    features = np.array(data['features'])
    print(features)
    
    # Make prediction
    prediction = model.predict(features.reshape(1, -1))
    
    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
