from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

application = Flask(__name__)
application.config['TEMPLATES_AUTO_RELOAD'] = True

@application.route('/', methods=['GET'])
def index():
  return render_template('cardio.html')

pickle_file = open('model.pickle', 'rb')
model = pickle.load(pickle_file)

@application.route('/predict', methods=['POST'])
def predict():
  user_data = request.json
  age, gender, height, weight, pulse = user_data['age'], user_data['gender'], user_data['height'], user_data['weight'], user_data['pulse']
  food, relative, smoke, heaviest = user_data['food'], user_data['relative'], user_data['smoke'], user_data['heaviest']
  race, pressure, salt, supps, tv = user_data['race'], user_data['pressure'], user_data['salt'], user_data['supps'], user_data['tv']
  age_smoke, milk, income = user_data['age_smoke'], user_data['milk'], user_data['income']
  risk = _model_prediction(income, pulse, age_smoke, race, height, weight, supps, food, milk, gender, smoke, salt, pressure, tv, relative, heaviest, age)
  return jsonify({'risk': risk})

def _model_prediction(income, pulse, age_smoke, race, height, weight, supps, food, milk, gender, smoke, salt, pressure, tv, relative, heaviest, age):
  income, tv, heaviest, bmi, race1, race2, race3, race4, race6, race7 = _clean_data(income, tv, heaviest, height, weight, race)
  X = np.array([income, pulse, age_smoke, race1, bmi, supps, food, race6, race4, race3, race2, milk, gender, race7, smoke, salt, pressure, tv, relative, heaviest, age]).reshape(1, -1)
  y_hat = model.predict_proba(X)
  print(y_hat)
  if y_hat[:,1] > 0.5:
    return 'HIGH-RISK'
  elif y_hat[:,1] <= 0.5:
    return 'NOT HIGH-RISK'

def _clean_data(income, tv, heaviest, height, weight, race):
  race_array = np.zeros(6)
  race_array[race-1] = 1
  race1, race2, race3, race4, race6, race7 = race_array
  poverty_ratio = income/20000
  tv = round(tv,1)
  heaviest = round(heaviest,1)
  bmi = (weight*0.453592)/(height*0.0254)**2
  return poverty_ratio, tv, heaviest, bmi, race1, race2, race3, race4, race6, race7

if __name__ == '__main__':
  application.run()