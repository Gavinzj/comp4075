# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:16:50 2018

@author: 60547
"""

from gensim.models import Word2Vec
 
from nltk.cluster import KMeansClusterer
import nltk
 
 
from sklearn import cluster
from sklearn import metrics
 
# training data
 
sentences = #[['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
            #['this', 'is',  'another', 'book'],
            #['one', 'more', 'book'],
            #['this', 'is', 'the', 'new', 'post'],
          #['this', 'is', 'about', 'machine', 'learning', 'post'],  
            #['and', 'this', 'is', 'the', 'last', 'post']]
           [['keyword']]
 
 
# training model
model = Word2Vec(sentences, min_count=1)
 
# get vector data
X = model[model.wv.vocab]

 
 
 
 
NUM_CLUSTERS=3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

 
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

words = list(model.wv.vocab)
for i, word in enumerate(words):  
    print (word + ":" + str(assigned_clusters[i]))
 
 
 
