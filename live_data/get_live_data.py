import requests
import socket
# really annoying to import a module from a dir above/sideways
# http://stackoverflow.com/questions/1054271/how-to-import-a-python-class-that-is-in-a-directory-above
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
import clean_data as cd
import models.gradient_boosting as gb

def get_live_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("gmail.com",80))
    my_ip = s.getsockname()[0]
    print 'ip is:', my_ip
    s.close()

    reg_url = 'http://10.6.80.211:5000/register'
    res = requests.post(reg_url, data={'ip': my_ip, 'port': 7001})
    return res

def get_test_data(path='../data/data.json', one_line=True):
    df = pd.read_json(path)
    if one_line:
        return df.iloc[0].to_json()

def make_one_prediction(json_data):
    preds = gb.unpickle_and_predict(json_data)

def test_prediction():
    json_data = get_test_data()
    make_one_prediction(json_data)
