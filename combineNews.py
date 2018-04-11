# -*- coding: utf-8 -*-
import nltk
import numpy as np
import lda
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
from nltk.corpus import stopwords

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
input_files = []

for dir_entry in os.listdir(path):
    dir_entry_path = os.path.join(path, dir_entry)
    if (os.path.isfile(dir_entry_path) and dir_entry_path.endswith("json")):
        input_files.append(dir_entry_path)
        print("Append file: "+dir_entry_path)
        

data = []
for input_file in input_files:
    with open(input_file,'r') as f:
        x = json.load(f)
        for item in x:
            if (len(item["articles"])>0):
                #print(str(len(item["articles"])))
                for i in range(len(item["articles"])):
                    pieces = (str(item["articles"][i]["description"]))
                    pieces = re.sub(r"http\S+","",str(pieces))
                    pieces = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", "",str(pieces))
                    pieces = re.sub('[^A-Za-z\s]', "", str(pieces))
                    #stopwords = ("a", "an", "the", "he", "she", "it")
                    stops = set(stopwords.words("english"))
                    pieces = str(pieces).lower()
                    tokens = str(pieces).split()
                    tokens = [w for w in tokens if w not in stops]
                    pieces = " ".join(tokens)
                    data.append(pieces)

#r = int(len(data)/1000)
#for i in range(r):
#    for piece in data[1000*i:1000*(i+1)-1]:
#        if piece:
#            sys.stdout=open("/Users/fzj/Desktop/comp4075/content"+str(i)+".txt","a+")
#            print("{}".format(piece))
#            sys.stdout.close()
            
r = int(len(data))
for i in range(r):
    piece = str(data[i])
    sys.stdout=open("/Users/fzj/Desktop/OnlineBTM/sample-data/"+str(i)+".txt","w")
    print("{}".format(piece))
    sys.stdout.close()
            
