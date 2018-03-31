import sys, os, inspect
import json
import re
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
from twitter_scraper import get_tweets
from stop_words import get_stop_words
import numpy as np



def getTwitterData(user, pages):
    en_stop = get_stop_words('english')
    words = ""
    for tweet in get_tweets(user, pages=pages):
        content = tweet['text']
        content = content.strip()
        content = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", " ", content)
        content = re.sub("[^A-Za-z\s]", " " ,content)
        content = content.lower()
        en_stop = tuple(en_stop)
        if en_stop is not None:
            tokens = content.split()
            tokens = [word for word in tokens if word not in en_stop]
            content = " ".join(tokens)
            words += content
    word_list = re.split('\ +', words)
    return word_list

print(getTwitterData('realDonaldTrump', 2))
