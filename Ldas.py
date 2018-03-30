import numpy as np
from sklearn.lda import LDA
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

input_files = ['/Users/fzj/Desktop/comp4075_project/news data/sources/output201803-23-28.json', '/Users/fzj/Desktop/comp4075_project/news data/sources/output20180329.json']
num_of_topocs = 300
n_top_words = 20
num_show_news = 300
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

	
print(len(data))
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

print(len(token_dict))

print("\n Build DTM")
tf = CountVectorizer(stop_words='english')

print("\n Fit DTM")
tfs1 = tf.fit_transform(token_dict.values())

#use lda
model = lda.LDA(n_topics=num_of_topocs, n_iter=5000, alpha=0.1, eta=0.01,random_state=10)

# we fit the DTM not the TFIDF to LDAz
print("\n Fit LDA to data set")
model.fit_transform(tfs1)

print("\n Obtain the words with high probabilities")
topic_word = model.topic_word_  # model.components_ also works

print("\n Obtain the feature names")
vocab = tf.get_feature_names()

topic_word = model.topic_word_  # model.components_ also works
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_
for i in range(num_show_news):
    print("news number{}: {} (top topic: {}) \n".format(i, titles[i], doc_topic[i].argmax()))
