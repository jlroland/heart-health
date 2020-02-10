from browser import document, ajax
import json

def get_input_data():
  age = document['age'].value
  gender = document['gender'].value
  height = document['height'].value
  weight = document['weight'].value
  pulse = document['pulse'].value
  frozen = document['frozen'].value
  relative = document['relative'].value
  income = document['income'].value
  smoke = document['smoke'].value
  heaviest = document['heaviest'].value
  return {'age': int(age),
          'gender': gender,
          'height': float("{0:.2f}".format(height)),
          'weight': float("{0:.2f}".format(weight)),
          'pulse': int(pulse),
          'frozen': int(frozen),
          'relative': relative,
          'income': income,
          'smoke': smoke,
          'heaviest': float("{0:.2f}".format(heaviest))}

def display_prediction(req):
  result = json.loads(req.text)
  document['prediction'].html = f"{result['risk']}"

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