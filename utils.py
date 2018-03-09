import sys, os 

import json
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import gensim
from nltk.tokenize import TweetTokenizer
import codecs
from sklearn import decomposition
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import numpy as np
from collections import Counter
import unicodedata
import random
import string

import re

# some basic functions

es_stop = get_stop_words('es') + ['https', 'http', 'htt',
                                    'ahh','pues', 'usted', 'ellos', 'el', 'ella', 'yo', 'tu',
                                    '.', ',', '...', 'com', 'co', 'jaja', 'zz'];
en_stop = get_stop_words('en');

raw_stop = ['wwwlas2orillas','www', 'http', '.com', 'ow.', '#', '@', 'googl', 'bitl', 'wpme']

def remove_accents(input_str):
     nfkd_form = unicodedata.normalize('NFKD', input_str)
     only_ascii = nfkd_form.encode('ASCII', 'ignore')
     return only_ascii

def is_number(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return False
    return True

def tokenizerx(doc):
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    raw =  doc.lower();
    raw = remove_accents(raw);
    tokens = tokenizer.tokenize(raw);
    clean_tokens = []
    for i in tokens:
        flag=0
        for j in raw_stop:
            if j in i: 
                flag+=1
        if flag==0: 
            if i: 
                # if not i in en_stop:
                clean_tokens.append(i) 
    clean_tokens = re.sub('['+string.punctuation+']', '', " ".join(clean_tokens)).split();
    return remove_accents(" ".join(clean_tokens))
