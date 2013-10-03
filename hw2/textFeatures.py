# -*- coding: utf-8 -*-
"""
Created on Thu Oct 03 11:08:06 2013

@author: ptmalmyr
"""

from operator import itemgetter
import numpy as np
import re

"""
Given a corpus returns features,
a list of tuples with ('word', frequency)
"""

def getFeatures(corpus):
    # regular expression to remove non-alphabetic characters
    sanitize = re.compile('[\W0-9]')
    wordDict = dict()
    
    for item in corpus:
        # sanitize the text        
        item = item.replace('\n','')
        itemText = sanitize.sub(' ',item)
        itemText = itemText.strip()
        
        for word in itemText.split():
            if word in wordDict:
                wordDict[word] = wordDict[word] + 1
            else:
                wordDict[word] = 1
        # end for word
    # end for item
    
    # sort by frequency for convenience   
    return sorted(wordDict.items(), key=itemgetter(1))

"""
Given a corpus and a word dictionary,
compute a feature vector for each example
"""
def vectorize(corpus, words):
    