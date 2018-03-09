import os
from flask import Flask, jsonify, request, send_from_directory
import fasttext as ft
import config as cn
from utils import tokenizerx

classifier = ft.load_model('/src/sentiment/model/tweets.model.bin')

def make_prediction(query):
    labels = classifier.predict_proba([query.lower()], k=2)[0]
    labels = map(lambda (k, v): {'tag': k.replace('__label__',"").replace('__',""), 'score': v}, labels)
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