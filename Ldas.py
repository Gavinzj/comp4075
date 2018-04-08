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

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
print("Open file: "+path)
input_files = []
num_of_topocs = 20
n_top_words = 200
num_iteration = 1000
num_news = 0

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
                    title = (str(item["articles"][i]["title"]) +"\t" + str(item["articles"][i]["url"]))
                    titles.append(title)


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
                    pieces = (item["articles"][i]["description"])
                    pieces = re.sub(r"http\S+","",str(pieces))
                    pieces = re.sub("[+\.\!\/_,$%^*(+\"\'@#]", "",str(pieces))
                    pieces = re.sub('[^A-Za-z\s]', "", str(pieces))
                    stopwords = ("a", "an", "the", "he", "she", "it")
                    pieces = str(pieces).lower()
                    tokens = str(pieces).split()
                    tokens = [w for w in tokens if w not in stopwords]
                    pieces = " ".join(tokens)
                    data.append(pieces)

	
print("number of data; "+str(len(data)))
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

print("number of words: "+str(len(token_dict)))

tf = CountVectorizer(stop_words='english')

tfs1 = tf.fit_transform(token_dict.values())

#use lda
model = lda.LDA(n_topics=num_of_topocs, n_iter=num_iteration, alpha=0.1, eta=0.01,random_state=10)

# we fit the DTM not the TFIDF to LDAz
model.fit_transform(tfs1)

print("\n Obtain the words with high probabilities")
topic_word = model.topic_word_  # model.components_ also works

print("\n Obtain the feature names")
vocab = tf.get_feature_names()

print("create title_topic.txt")
print("create topic.txt")

topic_word = model.topic_word_  # model.components_ also works
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "newsKeyword/title_topic.txt"),"a+")
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    sys.stdout.close()
    

doc_topic = model.doc_topic_
for i in range(num_news):
    sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "newsKeyword/topic.txt"),"a+")
    print("News number:{};Topic{};Descriptions:{};Topic{}; \n".format(i, titles[i], data[i], doc_topic[i].argmax()))
    sys.stdout.close()

num = 0    
for i in range(num_news):
    fileName = "newsGroup/Topic"+str(doc_topic[i].argmax())+"/"+str(num)+".txt";
    num = num + 1
    sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fileName),"w")
    print(data[i])
    sys.stdout.close()
    
