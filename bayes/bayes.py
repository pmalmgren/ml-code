# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Homework Assignment #2

This file contains the Bayes classifier functions
One computes the conditional probabilities
The other accepts novel examples and returns a prediction
wordDict 
"""
from scipy.stats import itemfreq
import numpy as np

def trainClassifier(spam,ham,wordDict):
	vPrior = lambda nc,n,p: (nc + 1/p)/(n+1)

	pSpam = float(len(spam))/(len(spam)+len(ham))
	pHam = float(len(ham))/(len(ham)+len(spam))

	spamTable = dict()
	hamTable = dict() 
	
	# because occurences of words in documents may form a distribution
	for item in wordDict:
		spamTable[item] = list()
		hamTable[item] = list()

	# for each word that occurs, build a list of each document it occured in
	# len(list) = number of documents the word appeared in
	# sum(list) = number of occurrences of the word in total (irrelevant)
	
	for example in spam:
		for feature in example.keys():
			spamTable[feature].append(example[feature])

	for example in ham:
		for feature in example.keys():
			hamTable[feature].append(example[feature])	

	# assume each word is equally likely
	p = float(len(wordDict))
	
	probTable = dict()

	for word in wordDict:
		probTable[word] = dict()
		
		if len(spamTable[word]) == 0: 
			spamBins = vPrior(np.zeros(np.shape(itemfreq(hamTable[word])[:,1]),dtype=np.float),float(len(spam)),p)
			hamBins = vPrior(itemfreq(hamTable[word])[:,1],float(len(ham)),p)
		elif len(hamTable[word]) == 0:
			hamBins = vPrior(np.zeros(np.shape(itemfreq(spamTable[word])[:,1]),dtype=np.float),float(len(ham)),p)
			spamBins = vPrior(itemfreq(spamTable[word])[:,1],float(len(spam)),p)
		else:
			spamFreq = itemfreq(spamTable[word])
			hamFreq = itemfreq(hamTable[word])
			binLength = max(len(spamFreq),len(hamFreq))
			hamBins = np.zeros((binLength,),dtype=np.float)
			spamBins = np.zeros((binLength,),dtype=np.float)
			hamBins[0:len(hamFreq)] = hamFreq[:,1]
			spamBins[0:len(spamFreq)] = spamFreq[:,1]
			hamBins = vPrior(hamBins,float(len(ham)),p)
			spamBins = vPrior(spamBins,float(len(spam)),p)	

		probTable[word]['spam'] = spamBins
		probTable[word]['ham'] = hamBins
		
	return (probTable,pSpam,pHam)

def classify(probTable,pSpam,pHam,example):
	
	for item in probTable:
		if item in example:
			if example[item] > len(probTable[item]['spam']):
				pSpam = pSpam * probTable[item]['spam'][-1]
				pHam = pHam * probTable[item]['ham'][-1]
			else:
				pSpam = pSpam * probTable[item]['spam'][example[item]-1]
				pHam = pHam * probTable[item]['ham'][example[item]-1]
		else:
			pSpam = pSpam * (1-sum(probTable[item]['spam']))
			pHam = pHam * (1-sum(probTable[item]['ham']))

	if pHam >= pSpam:
		return "ham"
	else:
		return "spam"
