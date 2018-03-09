from flask import Flask, jsonify, request, send_from_directory
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import json
import string

model_name = '/src/sentiment/model.hdf5'
model = load_model(model_name)
model._make_predict_function()

vocab_size = 10000
max_length = 200

def clean_tweet(tweet):
    return ' '.join(tweet.encode('utf-8').translate(None, string.punctuation).replace('\n',' ').split()).lower()

def predict(sentence=''):
    docs = [sentence]
    encoded_docs = [one_hot(d, vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    pred = model.predict([padded_docs])
    return {"neg": float(pred[0][0]), "pos": float(pred[0][1]) }


app = Flask(__name__)

@app.route('/')
def info():
    return 'Sentiment analysis api: <br> Usage: /predict/<sentence> <br> Returns a json file with the sentiment for the sentence'

@app.route('/predict/<sentence>')
def sentiment(sentence):
    response = predict( sentence=sentence )
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5001)