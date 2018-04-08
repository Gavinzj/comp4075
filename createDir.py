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

input_file = 'newsKeyword/topic.txt'
i = 0
with open(input_file,'r') as f:
        for line in f:
            print(line)
            contents = []
            contents = line.split(";")
            topic = contents[3]
            description = contents[2].split(":")[1]
            fileName = "newsGroup/"+topic+"/"+str(i)+".txt";
            i = i+1
            #sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fileName),"a+")
            #print("{}".format(description))
            #sys.stdout.close()
            
            file= open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fileName),"w")
            file.write(description)
            file.close()
            
            
