import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
import clean_data as cd
from flask import Flask, request, render_template
import json
import requests
import socket
import time
from datetime import datetime
import models.predict as prd
import pandas as pd
import json

app = Flask(__name__)
PORT = 5353
REGISTER_URL = "http://10.6.80.211:5000/register"
DATA = []
TIMESTAMP = []
predictions = []
frauds = []
current_predict = 1
predict_number = []


@app.route('/', methods=['GET'])
def homepage():
    return '''Welcome to our fraud detector!
        <form action="/check" method='GET' >
            <input type='submit' value='New data?' />
        </form>
        '''

@app.route('/score', methods=['POST'])
def score():
    DATA.append(json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ': ')))
    TIMESTAMP.append(time.time())
    data = DATA[-1]
    new_df = cd.prepare_test_dataframe(data)
    pred = model.predict_proba(new_df)
    predictions.append(pred[0][1])
    if pred[0][1] >= 0.09189:
        frauds.append(1)
    else:
        frauds.append(0)
    predict_number.append(current_predict)
    current_predict += 1
    return ""


@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}

# predict page
@app.route('/dashboard', methods=['GET'] )
def dashboard():
    fraud_bool = [bool(x) for x in frauds]
    df = pd.DataFrame({'Data Point': predict_number, 'Fraud Prediciton': fraud_bool})
    return '''<div> Total number of data points: {} </div> <div> Total number of frauds detected: {} </div>'''.format(len(DATA), len(frauds))



def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


if __name__ == '__main__':
    model = prd.load_model()
    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    print "attempting to register %s:%d" % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
