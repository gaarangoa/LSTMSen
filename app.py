import os
from flask import Flask, jsonify, request, send_from_directory
from utils import tokenizerx
import subprocess


cmds = ['/src/bin/fastText-0.1.0/fasttext', 'predict-prob', '/src/sentiment/model/tweets.model.bin', '-', '5']
interactive_model = subprocess.Popen( cmds, stdout=subprocess.PIPE, stdin=subprocess.PIPE )

def make_prediction(query):
    global interactive_model
    interactive_model.stdin.write(query + '\n')
    i = " ".join(interactive_model.stdout.readline().strip().split('__label__')).split()
    labels = {i[0]: float(i[1]), i[2]:float(i[3])}
    return labels

app = Flask(__name__)

@app.route('/')
def info():
    return 'Tag analysis api'

@app.route("/predict/<sentence>", methods=['GET', 'POST'])
def classify_text(sentence):
    response = make_prediction( tokenizerx( sentence.lower() ) )
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5001)