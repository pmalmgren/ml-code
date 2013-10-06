# -*- coding: utf-8 -*-
"""
Homework Assignment #2

Main file
"""
import bayes
import textFeatures
import random
import pickle
import time
import numpy as np

def main():
	with open('data/SMSSpamCollection') as input_file:
		text = input_file.read()
	text = text.strip()
	text = text.split('\n')

	# stop word cutoffs as per assignment
	stopWords = [10,25,50,100,500]

	# xSlice our data into five equal segments for fivefold cross validation
	# each segment has random indices
	indices = random.sample(xrange(len(text)),len(text))
	randomData = [text[i] for i in indices]
	stride = len(randomData)/5
	randomSlices = [[],[],[],[],[]]
	for i in range(1,len(randomData)-1,stride+1):
		randomSlices[i/stride] = (randomData[i-1:i+stride-1]) 	
	
	# iterate through all the xSlices and perform training/classification
	for xSlice in range(5):
		trainSet = list()
		testSet = randomSlices[xSlice]
		for i in range(5):
			if i == xSlice:
				continue
			else:
				trainSet = trainSet + randomSlices[i]

		baseDict = textFeatures.getFeatures(trainSet)
	
		# remove n most frequent words
		
		for cutoff in stopWords:
			wordDict = [baseDict[i][0] for i in range(0,len(baseDict)-cutoff)]	
			tp = 0
			fp = 0
			tn = 0
			fn = 0	
			# build feature vectors (not really, they're hash tables)
			trainSpam,trainHam = textFeatures.vectorize(trainSet,wordDict)
			testSpam,testHam = textFeatures.vectorize(testSet,wordDict)
   
			start = time.clock()
			probTable,pSpam,pHam = bayes.trainClassifier(trainSpam,trainHam,wordDict)	
			print "Train: %f" % (time.clock() - start)

			for item in testSpam:
				start = time.clock()
				prediction = (bayes.classify(probTable,pSpam,pHam,item))
				print "Test: %f" % (time.clock() - start)
				if prediction == 'spam':
					tp = tp + 1
				else:
					fn = fn + 1
			for item in testHam:
				prediction = (bayes.classify(probTable,pSpam,pHam,item))
				if prediction == 'ham':
					tn = tn + 1
				else:
					fp = fp + 1
			
			result = {'tp': tp, 'fp': fp, fn: 'fn', 'tn': tn}
			# write results to temporary file
			fName = 'output/expcutoff%dslice%d' % (cutoff,xSlice)
			#pickle.dump(result,open(fName,'w'))
if __name__ == '__main__':
	main()
