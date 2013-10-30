# -*- coding: utf-8 -*-
"""
main.py

reads in data, trains classifier using AdaBoost, then outputs a number of 
relevant statistics.

"""

import numpy as np
import random
from boost import *

def vectorize(data):
    vectorizedData = np.zeros((16,len(data)),dtype=np.float)    
    labels = np.zeros((len(data),),dtype=np.int)    
    
    for i in range(len(data)):
        example = data[i]
        example = example.split(',')
        if example[0] == 'republican':
            labels[i] = -1
        else:
            labels[i] = 1
            
        vector = np.zeros((16,),dtype=np.float)
        
        for j in range(16):
            if example[j+1] == 'y':
                vector[j] = 1
            elif example[j+1] == 'n':
                vector[j] = 0
            else:
                vector[j] = -1
        
        print vector
        vectorizedData[:,i] = vector
        print vectorizedData[:,i]
        
    return labels,vectorizedData
    

def main():
    with open('house-votes-84.data','r') as data_file:
        raw_data = data_file.read()
    raw_data = raw_data.strip()
    raw_data = raw_data.split('\n')
    labels,vectorData = vectorize(raw_data)
    perceptronAcc = np.zeros((10,),dtype=np.float)
    stumpAcc = np.zeros((10,),dtype=np.float)
    for i in range(10):    
        randomInds = random.sample(range(len(raw_data)),len(raw_data))
        fourth = int(len(raw_data)/4)    
        # train a model with a perceptron
        classifier = BoostModel('perceptron')
        classifier.train_perceptron(vectorData[:,randomInds[fourth*2:]],
                                    labels[randomInds[fourth*2:]],
                                    vectorData[:,randomInds[fourth:fourth*2]],
                                    labels[randomInds[fourth:fourth*2]])
        perceptronAcc[i] = classifier.evaluate(vectorData[:,randomInds[0:fourth]],
                                                labels[randomInds[0:fourth]])
        del classifier
        
    print "PERCEPTRON ACCURACY:"
    print perceptronAcc
    print "Average Error: %.4f +/- %.4f" % (np.mean(perceptronAcc),np.var(perceptronAcc))
    print "Max Error: %.4f Min Error: %.4f" % (np.max(perceptronAcc),np.min(perceptronAcc))
    
    
    for i in range(10):
        randomInds = random.sample(range(len(raw_data)),len(raw_data))
        fourth = int(len(raw_data)/4)    
        # train a model with a stump
        boostclassifier = BoostModel('stump')        
        boostclassifier.train_stump(vectorData[:,randomInds[fourth*2:]],
                                    labels[randomInds[fourth*2:]],
                                    vectorData[:,randomInds[fourth:fourth*2]],
                                    labels[randomInds[fourth:fourth*2]])
        stumpAcc[i] = boostclassifier.evaluate(vectorData[:,randomInds[0:fourth]],
                                                labels[randomInds[0:fourth]])
        del boostclassifier
    print "STUMP ACCURACY:"
    print stumpAcc
    print "Average Error: %.4f +/- %.4f" % (np.mean(stumpAcc),np.var(stumpAcc))
    print "Max Error: %.4f Min Error: %.4f" % (np.max(stumpAcc),np.min(stumpAcc))
    
if __name__ == '__main__':
    main()