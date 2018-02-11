from datetime import datetime
from requests import Request, Session
from pymongo import MongoClient
import time
from dateutil import parser
import json
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, Input, Reshape, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
import numpy as np
import sys
import re

from sklearn.preprocessing import StandardScaler

dataset = [i.split('\t') for i in open('dataset.tsv')]
# dataset has the structure sentence\tlabel for instance:
#   everything is fine  positive
#   everything is bad   negative

docs = [i[0] for i in dataset]
raw_labels = [i[1] for i in dataset]

vocab_size = 10000
max_length = 200
embedding_size = 1200

encoded_docs = [one_hot(d, vocab_size) for d in docs] #uses a hash function to represent words, if words are similar they will have collisions
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

labels_encoder = preprocessing.LabelEncoder()
labels_encoder.fit(raw_labels)
encoded_labels = labels_encoder.transform(raw_labels)
categorical_labels = np_utils.to_categorical( encoded_labels )

# Build the model
text_model_input = Input(shape = (max_length,), dtype="int32", name = 'text_model_input')
text_model = Embedding(input_dim = vocab_size, mask_zero=True, output_dim = embedding_size, input_length = max_length, name="text-embedding" )(text_model_input)
text_model = LSTM(256, name = "text-lstm-2", return_sequences=True)(text_model)
text_model_output = LSTM(256, name = "text-lstm-3", return_sequences=False)(text_model)

merged_model = Dense(1200, activation="relu")(text_model_output)
merged_model = Dropout(0.5)(merged_model)
merged_model = Dense(800, activation="relu")(merged_model)
merged_model = Dropout(0.5)(merged_model)
merged_model = Dense(600, activation="relu")(merged_model)
merged_model = Dense(200, activation="relu")(merged_model)
merged_model_output = Dense(2, activation = "softmax", name = 'merged_model_output')(merged_model)

model = Model(inputs = [text_model_input], outputs = [merged_model_output ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

try:
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
except:
    pass

# Train the model
checkpointer = ModelCheckpoint(filepath='./epoch/model.hdf5', verbose=1, save_weights_only=False)
model.fit([padded_docs], [categorical_labels], batch_size=800, epochs=100, callbacks=[checkpointer])

# save model
model.save('model.hdf5')