# -*- coding: utf-8 -*-
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition
import sklearn.feature_extraction.text as text
import xlsxwriter
import sys
import os
import requests

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
print("Open file: "+path)
input_files = []
num_of_topocs = 20
n_top_words = 400
num_show_news = 0

for dir_entry in os.listdir(path):
    dir_entry_path = os.path.join(path, dir_entry)
    if (os.path.isfile(dir_entry_path) and dir_entry_path.endswith("json")):
        input_files.append(dir_entry_path)
        print("Append file: "+dir_entry_path)
        
#load title
titles = []
for input_file in input_files:
    with open(input_file,'r') as f:
        x = json.load(f)
        for item in x:
            if (len(item["articles"])>0):
                for i in range(len(item["articles"])):
                    title = (str(item["articles"][i]["title"]) +"\t" + str(item["articles"][i]["url"])+"\t"+ str(item["articles"][i]["description"]))
                    titles.append(title)
num_show_news = len(titles)
tokenizer = RegexpTokenizer(r'\w+')
num_news = len(titles)
print("number of news: "+str(num_news))
# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# load news content
data = []
for input_file in input_files:
    with open(input_file,'r') as f:
        x = json.load(f)
        for item in x:
            if (len(item["articles"])>0):
                #print(str(len(item["articles"])))
                for i in range(len(item["articles"])):
                    pieces = (str(item["articles"][i]["title"])+" " + str(item["articles"][i]["description"]))
                    pieces = re.sub(r"http\S+","",str(pieces))
                    pieces = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", "",str(pieces))
                    pieces = re.sub('[^A-Za-z\s]', "", str(pieces))
                    stopwords = ("a", "an", "the", "he", "she", "it")
                    pieces = str(pieces).lower()
                    tokens = str(pieces).split()
                    tokens = [w for w in tokens if w not in stopwords]
                    pieces = " ".join(tokens)
                    data.append(pieces)
	
print("Total number of news: "+str(len(data)))
token_dict = {}
num = 0

for item in data:
    buf = item.split()
    # remove stop words from tokens
    stopped_tokens = [i for i in buf if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    buf = stemmed_tokens
    for i in range(len(buf)):
        token_dict[num] = buf[i]
        num = num+1

print("Total number of words: "+str(len(token_dict)))

#print("\n Build DTM")
vectorizer = text.CountVectorizer(stop_words='english', min_df=20)

dtm = vectorizer.fit_transform(titles).toarray()

vocab = np.array(vectorizer.get_feature_names())
print("Total number of vocabularies: "+str(len(vocab)))

clf = decomposition.NMF(n_components=num_of_topocs, random_state=1)
      
# we fit the DTM not the TFIDF to LDA
#print("\n Fit LDA to data set")
doctopic = clf.fit_transform(dtm)

topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:n_top_words]
    topic_words.append([vocab[i] for i in word_idx])

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

novel_names = []
for name in titles:
    name = name.rstrip('0123456789')
    novel_names.append(name)



novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()
num_groups = len(set(novel_names))
doctopic_grouped = np.zeros((num_groups, num_of_topocs))
for i, name in enumerate(sorted(set(novel_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

doctopic = doctopic_grouped
print("doc-topic matrix")
print(doctopic)

novels = sorted(set(novel_names))
print("create title_topic.txt")
print("create topic.txt")
print("doctopic.xlsx")
for i in range(len(doctopic)):
    top_topics = np.argsort(doctopic[i,:])[::-1][0:10]
    top_topics_str = ' '.join(str(t) for t in top_topics)
    sys.stdout=open("/Users/fzj/Desktop/comp4075/NMF/newsKeyword/title_topic.txt","a+")
    print("{}§ {}+\n".format(novels[i], top_topics_str))
    sys.stdout.close()
    

for t in range(len(topic_words)):
    sys.stdout=open("/Users/fzj/Desktop/comp4075/NMF/newsKeyword/topic.txt","a+")
    print("Topic {}: {}+\n".format(t, ' '.join(topic_words[t][:n_top_words])))
    sys.stdout.close()


workbook = xlsxwriter.Workbook('/Users/fzj/Desktop/comp4075/NMF/newsKeyword/doctopic.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(doctopic):
    worksheet.write_column(row, col, data)

workbook.close()

