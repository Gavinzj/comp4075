# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
import os
import numpy as np
import json
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition
import sklearn.feature_extraction.text as text
import xlsxwriter
import sys
import requests
import pickle
n_samples = 2000
n_features = 10000
n_components = 20
n_top_words = 200


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
input_files = []

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
                    title = (str(item["articles"][i]["title"]) +"§" + str(item["articles"][i]["url"])+"§"+ str(item["articles"][i]["description"]))
                    titles.append(title)
                    
print("number of titiles: "+str(len(titles)))
tokenizer = RegexpTokenizer(r'\w+')
num_news = len(titles)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

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
                    stops = set(stopwords.words("english"))
                    pieces = str(pieces).lower()
                    tokens = str(pieces).split()
                    tokens = [w for w in tokens if w not in stops]
                    pieces = " ".join(tokens)
                    data.append(pieces)
data_samples = data
print("Total number of news: "+str(len(data_samples)))
print("done in %0.3fs." % (time() - t0))

token_dict = {}
num = 0

for item in data:
    buf = item.split()
    # remove stop words from tokens
    stopped_tokens = [i for i in buf if not i in stops]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    buf = stemmed_tokens
    for i in range(len(buf)):
        token_dict[num] = buf[i]
        num = num+1


# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)

vectorizer = text.CountVectorizer(stop_words='english', min_df=20)
dtm = vectorizer.fit_transform(titles).toarray()
vocab = np.array(vectorizer.get_feature_names())
print("Total number of vocabularies: "+str(len(vocab)))
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()



# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

doctopic = nmf.fit_transform(dtm)

print("save model")
model_path = '/Users/fzj/Desktop/comp4075/NMF/finalized_model.sav'
pickle.dump(nmf, open(model_path, 'wb'))

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

novel_names = []
for name in titles:
    name = name.rstrip('0123456789')
    novel_names.append(name)
novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()
num_groups = len(set(novel_names))
doctopic_grouped = np.zeros((num_groups, n_components))
for i, name in enumerate(sorted(set(novel_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

doctopic = doctopic_grouped
print("doc-topic matrix")
print(doctopic)
novels = sorted(set(novel_names))
print("create title_topic.txt")

for i in range(len(doctopic)):
    top_topics = np.argsort(doctopic[i,:])[::-1][0:10]
    top_topics_str = ' '.join(str(t) for t in top_topics)

    orig_stdout = sys.stdout
    sys.stdout=open("/Users/fzj/Desktop/comp4075/NMF/newsKeyword/title_topic.txt","a+")
    print("{}§{}+\n".format(novels[i], top_topics_str))
    sys.stdout.close()
    sys.stdout=orig_stdout 


print("doctopic.xlsx")
workbook = xlsxwriter.Workbook('/Users/fzj/Desktop/comp4075/NMF/newsKeyword/doctopic.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(doctopic):
    worksheet.write_column(row, col, data)

workbook.close()

print("create topic.txt")
orig_stdout = sys.stdout
sys.stdout=open("/Users/fzj/Desktop/comp4075/NMF/newsKeyword/topic.txt","a+")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
sys.stdout.close()
sys.stdout=orig_stdout 


print("done in %0.3fs." % (time() - t0))
#
#
#
#
#
#
#
#
#
#
## Fit the NMF model
#print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
#      "tf-idf features, n_samples=%d and n_features=%d..."
#      % (n_samples, n_features))
#t0 = time()
#nmf = NMF(n_components=n_components, random_state=1,
#          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
#          l1_ratio=.5).fit(tfidf)
#print("done in %0.3fs." % (time() - t0))
#
#print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)










print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)

print("save model")
model_path = '/Users/fzj/Desktop/comp4075/Ldas/finalized_model.sav'
pickle.dump(lda, open(model_path, 'wb'))

print("\n Obtain the words with high probabilities")
topic_word = lda.components_  # model.components_ also works

print("\n Obtain the feature names")
vocab = tf_vectorizer.get_feature_names()


print("create topic.txt")
orig_stdout = sys.stdout
sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ldas/newsKeyword/topic.txt"),"a+")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
sys.stdout.close()
sys.stdout=orig_stdout

#print("create topic.txt")
#for i, topic_dist in enumerate(topic_word):
#    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
#    orig_stdout = sys.stdout
#    sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ldas/newsKeyword/topic.txt"),"a+")
#    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#    sys.stdout.close()
#    sys.stdout=orig_stdout 
    
print("create title_topic.txt")
doc_topic = lda.fit_transform(tf)
print(len(doc_topic))
for i in range(doc_topic.shape[0]):
    orig_stdout = sys.stdout
    sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ldas/newsKeyword/title_topic.txt"),"a+")
    print("{}§{} \n".format(titles[i], doc_topic[i].argmax()))
    sys.stdout.close()
    sys.stdout=orig_stdout 

    
doctopic = lda.fit_transform(dtm)

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

novel_names = []
for name in titles:
    name = name.rstrip('0123456789')
    novel_names.append(name)
novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()
num_groups = len(set(novel_names))
doctopic_grouped = np.zeros((num_groups, n_components))
for i, name in enumerate(sorted(set(novel_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

doctopic = doctopic_grouped
print("doc-topic matrix")
print(doctopic)

print("doctopic.xlsx")
workbook = xlsxwriter.Workbook('/Users/fzj/Desktop/comp4075/Ldas/newsKeyword/doctopic.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(doctopic):
    worksheet.write_column(row, col, data)

workbook.close()


print("done in %0.3fs." % (time() - t0))

