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
            print(a['description'])
            doclist.append(a['description'])
            print('\n') 