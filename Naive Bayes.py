#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd 
  
# reading csv file  
data = pd.read_csv("Data.csv")
data


# In[33]:


import random
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# In[ ]:





# In[39]:


from sklearn.model_selection import train_test_split

X=data[['SDNN', 'RMSSD']]  # Features
y=data['Result']


print(y)


# In[40]:


print(X)


# In[38]:


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
from sklearn import metrics


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[37]:


import math
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
import numpy as np
from sklearn.naive_bayes import GaussianNB


# In[ ]:


def calculateClassProbabilities(summaries, inputVector):
probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


# In[42]:


clf = GaussianNB()
clf.fit(X_train,y_train);
GaussianNB(priors=None, var_smoothing=1e-09)
a=clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, a))


# In[ ]:




