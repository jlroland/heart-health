from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return render_template('cardio.html')

@app.route('/predict', methods=['POST'])
def predict():
  user_data = request.json
  age, gender, height, weight, pulse = user_data['age'], user_data['gender'], user_data['height'], user_data['weight'], user_data['pulse']
  frozen, relative, income, smoke, heaviest = user_data['frozen'], user_data['relative'], user_data['income'], user_data['smoke'], user_data['heaviest']
  risk = _model_prediction(age, gender, height, weight, pulse, frozen, relative, income, smoke, heaviest)
  return jsonify({'risk': risk})

def _model_prediction(age, gender, height, weight, pulse, frozen, relative, income, smoke, heaviest):
  
  if y_pred==1:
    return 'high-risk'
  else:
    return 'low-risk'