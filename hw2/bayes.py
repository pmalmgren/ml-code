# -*- coding: utf-8 -*-
"""
Homework Assignment #2

This file contains the Bayes classifier functions
One computes the conditional probabilities
The other accepts novel examples and returns a prediction
704-786-9180
"""
import numpy as np

def trainClassifier(spam,ham,wordDict):
    prior = lambda nc,n,p: (nc + 1/p)/(n+1)

    jdt = np.zeros((len(wordDict),3),dtype=np.float)

    for example in 
