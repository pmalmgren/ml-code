# -*- coding: utf-8 -*-
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
	# weights for virtual examples
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
		for i in range(len(example)):
			if example[i] != 0:
				spamTable[wordDict[i]].append(example[i])

	for example in ham:
		for i in range(len(example)):
			if example[i] != 0:
				spamTable[wordDict[i]].append(example[i])

	# assume each word is equally likely
	p = float(len(wordDict))
	
	probTable = list()

	for i in range(len(wordDict)):
		word = wordDict[i]	
		if len(spamTable[wordDict[i]]) == 0: 
			spamBins = vPrior(np.zeros(np.shape(itemfreq(hamTable[word])[:,1]),dtype=np.float),float(len(spam)),p)
			hamBins = vPrior(itemfreq(hamTable[word])[:,1],float(len(ham)),p)
		elif len(hamTable[wordDict[i]]) == 0:
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

		ref = {'spam': spamBins, 'ham': hamBins}
		probTable.append(ref)
		
	return (probTable,pSpam,pHam)

def classify(probTable,pSpam,pHam,example):
	
	for i in range(len(probTable)):
		if example[i] != 0:
			if example[i] > len(probTable[i]['spam']):
				pSpam = pSpam * probTable[i]['spam'][-1]
				pHam = pHam * probTable[i]['ham'][-1]
			else:
				pSpam = pSpam * probTable[i]['spam'][example[i]-1]
				pHam = pHam * probTable[i]['ham'][example[i]-1]
		else:
			pSpam = pSpam * (1-sum(probTable[i]['spam']))
			pHam = pHam * (1-sum(probTable[i]['ham']))

	if pHam >= pSpam:
		return "ham"
	else:
		return "spam"
