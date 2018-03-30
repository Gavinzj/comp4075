import json
import re
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import sys

with open('sources/output201803-23-28.json', 'r') as f:
    x = json.load(f)
    desc = ""
    doclist = []
    ex2 = ""
    for item in x:
        for a in item['articles']:
            desc = a['description']
            desc = desc.strip()
            desc = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", " ", desc)
            desc = re.sub("[^A-Za-z\s]", " " ,desc)
            desc = desc.lower()
            doclist.append(desc)
            print(desc + "\n")