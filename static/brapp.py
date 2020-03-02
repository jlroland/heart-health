from browser import document, ajax
import json

def get_input_data():
  age = document['age'].value
  print(age)
  gender = 0
  print(gender)
  race = document['race'].value
  print(race)
  height = document['height'].value
  print(height)
  weight = document['weight'].value
  print(weight)
  pulse = document['pulse'].value
  print(pulse)
  heaviest = document['heaviest'].value
  print(heaviest)
  smoke = document['smoke'].value
  print(smoke)
  age_smoke = document['age_smoke'].value
  print(age_smoke)
  pressure = document['pressure'].value
  print(pressure)
  salt = document['salt'].value
  print(salt)
  relative = document['relative'].value
  print(relative)
  food = document['food'].value
  print(food)
  milk = document['milk'].value
  print(milk)
  supps = document['supps'].value
  print(supps)
  tv = document['tv'].value
  print(tv)
  income = document['income'].value
  print(income)
  
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