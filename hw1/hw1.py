#!/usr/bin/python2

import scipy as sp
import numpy as np

# Initialize data
# dt is a data container containing rows/labels

dt = np.dtype([('label',np.str_,1), ('data',np.int16, (16,))])

testData = np.zeros((5000,1),dt)
trainData = np.zeros((15000,1),dt)

counter = 0

# Read in data

with open('letter-recognition.data') as f:
    content = f.readline()
    content = content.split(',')
    
    label = content.pop(0)
    data = np.int16(content)

    if counter < 15000:
        trainData.put(counter,[(label,data)])
    else:
        testData.put(counter-15000,[(label,data)])
    
    counter = counter + 1
        
print "Success!"