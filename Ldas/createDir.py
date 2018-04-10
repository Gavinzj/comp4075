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

input_file = '/Users/fzj/Desktop/comp4075/Ldas/newsKeyword/title_topic.txt'
i = 0
with open(input_file,'r') as f:
        for line in f:
            #print(line)
            contents = []
            contents = line.split("\t")
            if(len(contents)>=4):
                name = contents[0]
                url = contents[1]
                description = contents[2]
                topic = "Topic"+str(contents[3])
                #print(topic)
                fileName = "/Users/fzj/Desktop/comp4075/Ldas/newsGroup/"+topic+"/"+str(i)+".txt";
                i = i+1
                sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fileName),"w")
                print("{}".format(description))
                sys.stdout.close()
            
            
