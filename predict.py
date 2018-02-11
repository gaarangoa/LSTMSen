from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import json

model_name = 'model.hdf5'
model = load_model(model_name)

vocab_size = 10000
max_length = 200


def predict(sentence=''):
    docs = [sentence]
    encoded_docs = [one_hot(d, vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    pred = model.predict([padded_docs])
    return pred

docs = "Hay personas que parecen que no estan haciendo nada ante los ojos de los demas, pero lo que no saben es que estan dando lo mejor que pueden. Denles  chance abrirse y entenderlos"
predict(docs)
