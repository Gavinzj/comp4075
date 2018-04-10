# -*- coding: utf-8 -*- 

from sklearn.linear_model import LogisticRegression   
#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import datasets
import numpy as np

train = datasets.load_files("train/")
test = datasets.load_files("test/")

#tokenizing text with sk-learn
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.data)

#tf–idf can be computed as follows:
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#train classifier
clf = MultinomialNB().fit(X_train_tfidf, train.target)

classifier = LogisticRegression()  
classifier.fit(X_train_tfidf, train.target)   

#Prepare the testing data set
X_test_counts = count_vect.transform(test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts) 

#use the trained classifier to predict results for testing data set
predicted = clf.predict(X_test_tfidf)
count=0.0
for doc, category in zip(test.data, predicted):
  if(train.target_names[category]=="Topic0"):
    #print("***")
    count+=1  
#print(count)
result = (count/100.0)
#print('Classified as: %s\n%s\n' % (train.target_names[category],doc))
print(result)