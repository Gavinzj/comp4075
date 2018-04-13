# -*- coding: utf-8 -*-
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
from nltk.corpus import stopwords


user = "realDonaldTrump"
pages = 5

fileName = "/Users/fzj/Desktop/comp4075/OnlineBTM/sample-data/0.txt"
for tweet in get_tweets(user, pages=pages):
    content = tweet['text']
    content = re.sub(r"http\S+","",str(content))
    content = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", "",str(content))
    content = re.sub('[^A-Za-z\s]', "", str(content))
    stops = set(stopwords.words("english"))
    content = str(content).lower()
    tokens = str(content).split()
    tokens = [w for w in tokens if w not in stops]
    output = ""
    for token in tokens:
        output = output +" " +token
    orig_stdout = sys.stdout
    sys.stdout=open(fileName,"a+")
    print("{}".format(output))
    sys.stdout.close()
    sys.stdout=orig_stdout
    