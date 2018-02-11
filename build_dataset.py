import pytz
from datetime import datetime
from requests import Request, Session
from pymongo import MongoClient
import time
from dateutil import parser
import json
import numpy as np
import sys
import re

from sklearn.preprocessing import StandardScaler
import nltk
import string

port = int(sys.argv[1]) #port where the mongo client is connected

# I store my data in a mongo db container. 
client = MongoClient("mongodb://localhost", port)
raw_neg_1 = [i for i in client.twitter['%3Aenojado%20triste&l=es'].find()[1:] if not '@' in i['text']]
raw_neg_2 = [i for i in client.twitter['%3Anegativo&l=es'].find()[1:] if not '@' in i['text']]

raw_neg = raw_neg_1 + raw_neg_2
raw_pos_1 = [i for i in client.twitter['%3Afeliz&l=es'].find()[1:] if not '@' in i['text']]
raw_pos_2 = [i for i in client.twitter['%3Apositivo&l=es'].find()[1:] if not '@' in i['text']]

raw_pos = raw_pos_1 + raw_pos_2

def clean_tweet(tweet):
    return ' '.join(tweet.encode('utf-8').translate(None, string.punctuation).replace('\n',' ').split()).lower()

_is_clean = {}

pos = []
for i in raw_pos:
    try:
        x = _is_clean[i['text']]
    except Exception as e:
        _is_clean.update({i['text']: True})
        try:
            pos.append(clean_tweet(i['text']))
        except Exception as e:
            pass


_is_clean = {}  
neg = []
for i in raw_neg:
    if 'no prometas' in i['text'].lower(): continue
    try:
        x = _is_clean[i['text']]
    except Exception as e:
        _is_clean.update({i['text']: True})
        try:
            neg.append(clean_tweet(i['text']))
        except Exception as e:
            pass

print("positive: ", len(pos), "negative: ", len(neg))
fo = open('dataset.tsv', 'w')

_min = min([len(pos), len(neg)])

for i in pos[:_min]:
    fo.write(i.replace('positivo', ' ')+'\tpositive\n')

for i in neg[:_min]:
    fo.write(i.replace('negativo', ' ')+'\tnegative\n')

print("positive: ", len(pos), "negative: ", len(neg))

fo.close()




