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
  frozen, relative, smoke, heaviest = user_data['frozen'], user_data['relative'], user_data['smoke'], user_data['heaviest']
  risk = _model_prediction(age, gender, height, weight, pulse, frozen, relative, smoke, heaviest)
  return jsonify({'risk': risk})

def _model_prediction(age, gender, height, weight, pulse, frozen, relative, smoke, heaviest):
  gender, bmi, relative, smoke = _clean_data(gender, height, weight, relative, smoke)
  X = np.array([age, gender, bmi, pulse, frozen, relative, smoke, heaviest]).reshape(1, -1)
  scaled = MinMaxScaler()
  X_scaled = scaled.fit_transform(X)
  y_hat = model.predict_proba(X_scaled)[:,1]
  if y_hat > 0.5:
    return 'high-risk'
  else:
    return 'low-risk'

def _clean_data(height, weight, relative, smoke):
  relative, income, smoke = gender.lower(), relative.lower(), smoke.lower()
#   if gender=='m':
#     gender=1
#   else:
#     gender=0
  if relative=='y':
    relative=1
  else:
    relative=0
  if smoke=='y':
    smoke=1
  else:
    smoke=0
  bmi = (weight*0.453592)/(height*0.0254)**2
  return bmi, relative, smoke

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)