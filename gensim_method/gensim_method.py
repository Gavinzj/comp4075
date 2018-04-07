# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
import logging
import re
from collections import defaultdict  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#collect the documents that used to compare
documents=[]
for i in range(0,19):
    content=""
    for line in open("topic"+str(i)+".txt"):
        content += line
    documents.append(content)

#1.devide the words by space
texts=[[word for word in document.lower().split() ] for document in documents]

#2.calculate the tfidf value
#create a dictionary
frequency = defaultdict(int) 
#calculate the word frequncey
for text in texts:	
    for token in text:		
        frequency[token]+=1
        #we only process the words appears at least twice
texts=[[token for token in text if frequency[token]>1] for text in texts]

#3.create a dictionary
dictionary=corpora.Dictionary(texts)

#4.turn the hashtags into a vector
new_doc=""
for line in open("sport.txt"):
        new_doc += line

new_vec = dictionary.doc2bow(new_doc.lower().split())

#5.建立语料库#将每一篇文档转换为向量
corpus = [dictionary.doc2bow(text) for text in texts]

#6.initialize the model
tfidf = models.TfidfModel(corpus)
#test
test_doc_bow = [(0, 1), (1, 1)]

#将整个语料库转为tfidf表示方法
corpus_tfidf = tfidf[corpus]

#7.create index
index = similarities.MatrixSimilarity(corpus_tfidf)

#8.calculate the similarity
new_vec_tfidf=tfidf[new_vec]

#calculate the similarities between the hashtag and each topics
sims = index[new_vec_tfidf]
result=0
max=0
for i in range(0,len(sims)):
    if(sims[i]>max):
        max=sims[i]
        result=i
print result,sims[result]
