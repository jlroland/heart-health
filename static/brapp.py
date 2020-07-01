from browser import document, ajax
import json

def get_input_data():
  age = document['age'].value
  gender = 0
  race = document['race'].value
  height = document['height'].value
  weight = document['weight'].value
  pulse = document['pulse'].value
  heaviest = document['heaviest'].value
  smoke = document['smoke'].value
  age_smoke = document['age_smoke'].value
  pressure = document['pressure'].value
  salt = document['salt'].value
  relative = document['relative'].value
  food = document['food'].value
  milk = document['milk'].value
  supps = document['supps'].value
  tv = document['tv'].value
  income = document['income'].value
  
  return {'age': int(age),
          'gender': gender,
          'race': int(race),
          'height': float(height),
          'weight': float(weight),
          'pulse': int(pulse),
          'heaviest': float(heaviest),
          'smoke': smoke,
          'age_smoke': int(age_smoke),
          'pressure': pressure,
          'salt': salt,
          'relative': relative,
          'food': int(food),
          'milk': milk,
          'supps': int(supps),
          'tv': float(tv),
          'income': float(income),
          }

def display_prediction(req):
  result = json.loads(req.text)
  document['prediction'].html = result['risk']

def send_input_data(data):
  req = ajax.Ajax()
  req.bind('complete', display_prediction)
  req.open('POST', '/predict', True)
  req.set_header('Content-Type', 'application/json')
  req.send(json.dumps(data))


def click(event):
  user_info = get_input_data()
  send_input_data(user_info)

document['predict'].bind('click', click)