from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return render_template('cardio.html')

@app.route('', methods=['POST'])
def predict():
  user_data = request.json
  #deconstruct object data into usable variables
  #define return variables by invoking function for model
  return jsonify({}) #return variables with values as dictionary