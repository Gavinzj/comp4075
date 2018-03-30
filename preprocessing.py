import json
import re
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np

with open('sources/output20180329.json', 'r') as f:
    x = json.load(f)
    content = ""
    doclist = []
    ex2 = ""
    for item in x:
        for a in item['articles']:
            content = a['description']
            content = content.strip()
            content = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", " ", content)
            content = re.sub("[^A-Za-z\s]", " " ,content)
            content = content.lower()
            doclist.append(" " + content)
            print(content)
            print('\n') 