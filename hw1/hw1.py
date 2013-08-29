#!/usr/bin/python2

import scipy as sp
import numpy as np

# Initialize data

trainingExamples = np.zeros((15000,16),dtype=np.int8)
trainingLabels = []

testExamples = np.zeros((5000,16),dtype=np.int16)
testLabels = []

counter = 0

# Read in data

with open('letter-recognition.data') as f:
    content = f.readline()
    content = content.split(',')
    
    label = content.pop()
    data = np.array(content,dtype=np.int16)    

    if counter <= 14999:
        trainingExamples[counter,:] = data
        trainingLabels[counter] = label
    else:
        testExamples[counter-15000,:] = data
        testLabels[counter] = label;
        
