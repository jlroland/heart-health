from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return render_template('cardio.html')

pickle_file = open('model.pickle', 'rb')
model = pickle.load(pickle_file)

@app.route('/predict', methods=['POST'])
def predict():
  user_data = request.json
  age, gender, height, weight, pulse = user_data['age'], user_data['gender'], user_data['height'], user_data['weight'], user_data['pulse']
  food, relative, smoke, heaviest = user_data['food'], user_data['relative'], user_data['smoke'], user_data['heaviest']
  race, pressure, salt, supps, tv, income = user_data['race'], user_data['pressure'], user_data['salt'], user_data['supps'], user_data['tv'], user_data['income']
  risk = _model_prediction(income, pulse, 'SMD030', 'race_1.0', height, weight, supps, food, 'race_6.0',
       'race_4.0', 'race_3.0', 'race_2.0', 'milk_0.0', gender, 'race_7.0', 'SMQ020', salt, pressure,
       'PAQ710', relative, heaviest, age)
  return jsonify({'risk': risk})

def _model_prediction(age, gender, height, weight, pulse, frozen, relative, smoke, heaviest):
  bmi = _clean_data(height, weight)
  X = np.array([age, gender, bmi, pulse, frozen, relative, smoke, heaviest]).reshape(1, -1)
  scaled = MinMaxScaler()
  X_scaled = scaled.fit_transform(X)
  y_hat = model.predict_proba(X_scaled)[:,1]
  if y_hat > 0.5:
    return 'high-risk'
  else:
    return 'low-risk'

def _clean_data(height, weight):
  
  bmi = (weight*0.453592)/(height*0.0254)**2
  return bmi

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)