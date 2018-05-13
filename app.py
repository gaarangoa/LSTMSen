import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from utils import tokenizerx
import subprocess
import re
import unicodedata

cmds = ['/src/bin/fastText-0.1.0/fasttext', 'predict-prob',
        '/src/sentiment/model/tweets.model.bin', '-', '5']
interactive_model = subprocess.Popen(
    cmds, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


def clean_tweet(text):
    txt = ' '.join(re.sub(
        "(#[A-Za-z0-9]+)|(@[A-Za-z0-9_]+)|([\.\/\,\:\?\!\;])|(\w+:\/\/\S+)", " ", text.lower()).split())
    try:
        txt = unicode(txt, 'utf-8')
    except (TypeError, NameError):  # unicode is a default on python 3
        pass
    txt = unicodedata.normalize('NFD', txt)
    txt = txt.encode('ascii', 'ignore')
    txt = txt.decode("utf-8")
    txt = re.sub('[ ]+', ' ', txt)
    txt = re.sub('[^0-9a-zA-Z_-]', ' ', txt)
    txt = re.sub(r'([a-z][a-z])\1+', r'\1', txt)
    return txt


def make_prediction(query):
    global interactive_model
    interactive_model.stdin.write(clean_tweet(query) + '\n')
    i = " ".join(interactive_model.stdout.readline(
    ).strip().split('__label__')).split()
    labels = {i[0]: float(i[1]), i[2]: float(i[3])}
    return labels


app = Flask(__name__)
CORS(app)


@app.route('/')
def info():
    return 'Tag analysis api'


@app.route("/predict/", methods=['GET', 'POST'])
def classify_text():
    data = request.get_json()
    sentence = data['sentence']
    response = make_prediction(tokenizerx(sentence.lower()))
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
