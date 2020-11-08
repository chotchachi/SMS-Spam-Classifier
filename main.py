# This is a sample Python script.
# ta nhận thấy trong tiếng việt thường có những dấu ở các chữ cái trên  => thay thế chữ cái

# xóa các ký tự đặc biệt và dấu trong message tiền xử lý văn bản

#mark: load corpus txt

#mark: load stopword
from sklearn.svm import LinearSVC
import Utils
import pickle
import time
from flask import request
from urllib.parse import urlparse,parse_qs
# Press the green button in the gutter to run the script.
from flask import Flask
app = Flask(__name__)

def predict(message):
    svm = pickle.load(open('svm', 'rb'))
    bag_word = pickle.load(open('data', 'rb'))
    return  svm.predict([Utils.handleMessage(message, bag_word)])[0]

@app.route('/hello')
def hello_world():
  return 'Hello, World!'

@app.route('/message')
def message():
    try:
        message = request.args.get('text', type=str)
        isSpam = predict(str(message))
        if (isSpam == 1):
            return 'Spam'
        else:
            return 'Ham'
        return message
    except:
        return 'This page does not exist', 404



HOST_NAME = 'localhost'
PORT = 7979
if __name__ == '__main__':
  app.run()

