# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Homework Assignment #2

Classifies examples using a Naive Bayes Classifier
"""
import bayes
import textFeatures
import random
import pickle

def printResults(results):
    specificity = []
    sensitivity = []
    precision = []
    recall = []
    tpRate = []
    fpRate = []
    error = []
    accuracy = []

    for data in results:
        accuracy.append((data['tp']+data['tn'])/(data['tn']+data['tp']+data['fp']+data['fn']))
        error.append(1-accuracy[-1])
        fpRate.append((data['fp']/(data['fp']+data['tn'])))
        tpRate.append((data['tp']/(data['tp']+data['fn'])))
        recall.append(tpRate[-1])
        precision.append(data['tp']/(data['tp']+data['fp']))
        sensitivity.append(tpRate[-1])
        specificity.append(data['tn']/(data['fp']+data['tn']))

    print "==== Experiment Results (average values) ===="
    print "Specificity: %.4f" % (float(sum(specificity))/float(len(specificity)))
    print "Sensitivity: %.4f" % (float(sum(sensitivity))/float(len(sensitivity)))
    print "Precision: %.4f" % (float(sum(precision))/float(len(precision)))
    print "Recall: %.4f" % (float(sum(recall))/float(len(recall)))
    print "TP Rate: %.4f" % (float(sum(tpRate))/float(len(tpRate)))
    print "FP Rate: %.4f" % (float(sum(fpRate))/float(len(fpRate)))
    print "Error: %.4f" % (float(sum(error))/float(len(error)))
    print "Accuracy: %.4f" % (float(sum(accuracy))/float(len(accuracy)))
 

def main():
    with open('data/SMSSpamCollection') as input_file:
        text = input_file.read()
    text = text.strip()
    text = text.split('\n')

    cutoff = raw_input("How many words to truncate from dictionary: ")
    try:
        cutoff = int(cutoff)
    except:
        print "Invalid input, defaulting to 10"
        cutoff = 10
    
    # slice our data into five equal segments for fivefold cross validation
    # each segment has random indices
    indices = random.sample(xrange(len(text)),len(text))
    randomData = [text[i] for i in indices]
    stride = len(randomData)/5
    randomSlices = [[],[],[],[],[]]
    for i in range(1,len(randomData)-1,stride+1):
        randomSlices[i/stride] = (randomData[i-1:i+stride-1])

    print "Entering 'n' will use 1/5th of data for testing, the rest for training"
    print "Entering 'y' will use full cross validation and may take a while"
    crossValidate = raw_input("Perform cross validation? (y/n): ")
    
    if crossValidate.lower() != 'y' and crossValidate.lower() != 'n':
        print "Invalid input, not using cross validation"
        limit = 1
    elif crossValidate.lower() == 'y':
        limit = 5
    else:
        limit = 1
    
    resultList = list()
    for xSlice in range(limit):
        trainSet = list()
        testSet = randomSlices[xSlice]
        for i in range(5):
            if i == xSlice:
                continue
            else:
                trainSet = trainSet + randomSlices[i]

        print "Building dictionary..."
        baseDict = textFeatures.getFeatures(trainSet)
        wordDict = set([baseDict[i][0] for i in range(0,len(baseDict)-cutoff)])

        print "Vectorizing documents..."
        trainSpam,trainHam = textFeatures.vectorize(trainSet,wordDict)
        testSpam,testHam = textFeatures.vectorize(testSet,wordDict)

        print "Training classifier..."
        probTable,pSpam,pHam = bayes.trainClassifier(trainSpam,trainHam,wordDict)

        tp,fp,tn,fn = 0.0,0.0,0.0,0.0
        print "Beginning testing..."
        
        total = len(testSpam) + len(testHam)
        count = 0
        for item in testSpam:
            prediction = bayes.classify(probTable,pSpam,pHam,item)
            if prediction == 'spam':
                tp = tp + 1.0
            else:
                fp = fp + 1.0
            count = count + 1
            if count % 50 == 0:
                print "%d/%d complete" % (count,total)
        for item in testHam:
            prediction = bayes.classify(probTable,pSpam,pHam,item)
            if prediction == 'ham':
                tn = tn + 1.0
            else:
                fp = fp + 1.0
            count = count + 1
            if count % 50 == 0:
                print "%d/%d complete" % (count,total)
        print "Finished testing.."

        result = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        resultList.append(result)

    printResults(resultList)

if __name__ == '__main__':
    main()
