#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import re
import string
import csv
from collections import OrderedDict
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# In[2]:


# The following class and two functions have been taken from wikipedia at https://en.wikipedia.org/wiki/Trie#Algorithms

class Node():
    def __init__(self):
       # Note that using dictionary for children (as in this implementation) would not allow lexicographic sorting mentioned in the next section (Sorting),
       # because ordinary dictionary would not preserve the order of the keys
        self.children = {}  # mapping from character ==> Node
        self.value = None

def find(node, key):
    for char in key:
        if char in node.children:
            node = node.children[char]
        else:
            return None
    return node.value
    
def insert(root, string, value):
    node = root
    index_last_char = None
    for index_char, char in enumerate(string):
        if char in node.children:
            node = node.children[char]
        else:
            index_last_char = index_char
            break

            # append new nodes for the remaining characters, if any
    if index_last_char is not None: 
        for char in string[index_last_char:]:
            node.children[char] = Node()
            node = node.children[char]

    # store value in the terminal node
    node.value = value

# The following two functions have been written by the programmers for additional purposes of the trie    
    
def find_multiple(node, keys):
    # Return values for multiple Keys in the trie Node in order that keys are presented
    holder = node
    vals = [None]*len(keys)
    counter = 0
    for key in keys:
        node = holder
        for char in key:
            if char in node.children:
                node = node.children[char]
        vals[counter] = node.value
        counter += 1
    return vals

def update(node, key, difference):
    # Change the value which is currently stored for the Key in the trie Node by a value of Difference
    for char in key:
        if char in node.children:
            node = node.children[char]
    node.value += difference


# In[3]:


# the sets are a data structure which was solely used for checking results with the trie
listtopics=set()
listplaces=set()
listwords = set()
article_tries = [None]*21578
articleTopics = [None]*21578
articlePlaces = [None]*21578
# counts for no topics and no places in articles
cntnotop=0 
cntnoplc=0 
# Trie for the different topics, locations, and count of every word present across all articles
trieTopics = Node()
trieLoc = Node()
WordCount = Node()
# csv to hold all word values per article (row)

articleCount = 0


# In[4]:




for i in range(0,22):
    # over all files
    if(i>=10):
        # file names differ by the #, which is double digit for i>=10
        filename = 'reut2-0'+str(i)+'.sgm'
    else:
        filename = 'reut2-00'+str(i)+'.sgm'
    path = '' + filename
    file = open(path, 'rb')
    data = file.read()
    x = re.findall(r'<REUTERS(.*?)</REUTERS>', data.decode("windows-1252"), re.DOTALL)
    # finds all instances of "<REUTERS . . ." in a given file and save them 
    
    for j in range(0,len(x)):
        # for all articles in a file since every article starts with the REUTERS tag
        yTopic = re.findall(r'<TOPICS>(.*?)</TOPICS>', x[j], re.DOTALL)
        # store all topics in an article since an article can have multiple topics
        
        for k in range(0,len(yTopic)):
            lt = yTopic[k]

            article_topics = []
            topics = re.findall(r'<D>(.*?)</D>', lt, re.DOTALL)
            # Make sure D tag does not included as part of the topic name
            if(len(topics)==0):
                # length is 0 when there is no topic
                cntnotop=cntnotop+1
            for l in topics:
                # for every topic found in an article
                if (find(trieTopics,l) == None):
                    # check if the topic is already in the trie, and if not insert it with value 1
                    insert(trieTopics, l, 1)
          #          insert(article_topics, l, 1)
           #     elif (find(article_topics, l) == None):
            #        insert(article_topics, l, 1)
             #       update(trieTopics, l, 1)
                else:
                    # its been found already in the trie so increase the value by 1
                    update(trieTopics, l, 1)
              #      update(article_topics, l, 1)
                article_topics.append(l)
                listtopics.add(l)
        
        article_places = []
        yPlace = re.findall(r'<PLACES>(.*?)</PLACES>', x[j], re.DOTALL)
        for k in range(0,len(yPlace)):
            lt = yPlace[k]
            places = re.findall(r'<D>(.*?)</D>', lt, re.DOTALL)
            if(len(places)==0):
                cntnoplc=cntnoplc+1
            for l in places:
                if (find(trieLoc, l) == None):
                    insert(trieLoc, l, 1)
                else:
                    update(trieLoc, l, 1)
                article_places.append(l)
                listplaces.add(l)

        article_words = Node()        
        yBody = re.findall(r'<BODY>(.*?)</BODY>', x[j], re.DOTALL)
        for b,word in enumerate(yBody):
            # split the body into a bunch of different words
            body = word.split()
            body = [element.lower() for element in body] ; body            
            for l in body:
                if (find(WordCount, l) == None):
                    insert(WordCount, l, 1)
                    insert(article_words, l, 1)
                elif (find(article_words, l) == None):
                    insert(article_words, l, 1)
                    update(WordCount, l, 1)
                else:
                    update(WordCount, l, 1)
                    update(article_words, l, 1)
                listwords.add(l)
        #print (article_topics)
        #article_tries[articleCount] = [article_words]
        articlePlaces[articleCount] = article_places
        if(article_words == None):
                print ("found you")
        articleCount += 1

        
# end of main for loop for all files

# Print statements for the distinct list of topics, distinct list of places, and counts of topic-less and/or place-less 
# articles.  Although, we used set data structures to display the different keys here, it is easy to fetch values for keys 
# using a trie displayed below each.  Usage of sets was only done as part of "developing our domain-specific knowledge".
listtopics = list(listtopics)
listplaces = list(listplaces)
listwords = list(listwords)
#print(listtopics)
#print(find_multiple(trieTopics, listtopics))
#print(listplaces)
#print(find_multiple(trieLoc, listplaces))
#print("Data objects with no entries for topics: " + str(cntnotop))
#print("Data objects with no entries for places: " + str(cntnoplc))


# In[5]:


# Some tests for "finds" on the tries are shown below

#print (find(trieLoc, "usa"))


# In[ ]:


'''
print(find(article_tries[10][0], 'a'))
for i in range(len(article_tries)):
    article_tries[i] = find_multiple(article_tries[i][0], listwords)
article_tries =[i if i[0] is not None else (0, i[1]) for i in article_tries]
print((article_tries[1]))
#print(np.dtype(article_tries[1][0]))
np.savetxt("article_tries.csv", article_tries, delimiter=",", fmt='%s')
'''
np.savetxt("article_places.csv", articlePlaces, delimiter=",", fmt='%s')

# In[ ]:



# In[ ]:


print(article_tries)


# In[ ]:


print (find_multiple(trieLoc, ['usa', 'west-germany']))


# In[ ]:


print (find(WordCount, 'agriculture'))
print (find(WordCount, 'a'))


# In[ ]:


print (len(listtopics) ) #120
trieVals = find_multiple(trieTopics, listtopics)
#print (trieVals)
#topictrie[0] = listtopics
#topictrie[1] = trieVals
#print (topictrie[1])
'''with open('output_trie_topics.csv', 'w') as csvfile:
    fieldnames = ['topic', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(listtopics)):
        writer.writerow({'topic': topictrie[0][i], 'value': topictrie[1][i]})'''
#np.savetxt("output_trei_topics.csv", topictrie, delimiter=",")
#print (len(find_multiple(article_tries[0][2], listwords)))


# In[ ]:


# DBSCAN code

db = DBSCAN(eps=0.3, min_samples=10)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()