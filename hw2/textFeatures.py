# -*- coding: utf-8 -*-
"""
Homework Assignment #2

This file contains functions to build a vocabulary from a corpus,
and vectors from a corpus given a vocabulary.

"""

from operator import itemgetter
import re

"""
Given a corpus returns features,
a list of tuples with ('word', frequency)
in ascending order
"""

def getFeatures(corpus):
    # regular expression to remove non-alphabetic characters
    sanitize = re.compile('[\W0-9]')
    wordDict = dict()

    for item in corpus:
        if len(item) == 0:
            continue
        item = item.split('\t')
        text = item[1]
        # sanitize the text
        text = sanitize.sub(' ',text)
        text = text.strip()
        text = text.lower()
        text = text.split()

        for word in zip(text,text[1:]):
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
compute a feature vector for each example class
"""
def vectorize(corpus, wordDict):
    sanitize = re.compile('[\W0-9]')

    spam = []
    ham = []

    for item in corpus:
        if len(item) == 0:
            continue
        feats = dict()
        item = item.split('\t')
        tag = item[0]
        text = item[1]
        text = sanitize.sub(' ',text)
        text = text.strip()
        text = text.lower()
        text = text.split()

        for bigram in zip(text,text[1:]):
            word = bigram[0] + ',' + bigram[1]
            if word in wordDict:
                if word in feats:
                    feats[word] = feats[word] + 1
                else:
                    feats[word] = 1
                # end if word in feats
            # end if word in dictionary
        # end for word in text

        if tag == 'ham':
            ham.append(feats)
        elif tag == 'spam':
            spam.append(feats)
        else:
            print "Error: %s" % (tag)
    # end for item in corpus

    return (spam,ham)
