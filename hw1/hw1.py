#!/usr/bin/python2

import numpy as np
import dist
import time
import random

def main():
    # Initialize data
    # dt is a data container containing rows/labels
    
    dt = np.dtype([('labels',np.str_, 1), ('data',np.int16, (16,))])
    
    testData = np.zeros((5000,1),dt)
    trainData = np.zeros((15000,1),dt)
    
    counter = 0
    
    # Read in data
    
    with open('letter-recognition.data') as f:
        for line in f:
            content = line.split(',')
            
            label = content.pop(0)
            data = np.int16(content)
            
            if counter < 15000:
                trainData.put(counter,[(label,data)])
            else:
                testData.put(counter-15000,[(label,data)])
            
            counter = counter + 1
         # end for line in f
    # end with open
    
    n = 9000
    
    sampleData = trainData[random.sample(range(15000),n)]
    
    t1 = time.time()
    dist.knn(7,sampleData,testData[0])
    t2 = time.time()
    print '%s took %0.3f ms' % ('knn', (t2-t1)*1000.0)    
    
    t1 = time.time()
    minSet = dist.condense(trainData)
    t2 = time.time()
    print '%s took %0.3f ms' % ('condensing', (t2-t1)*1000.0) 

    print "minset length: %d \n" % (len(minSet))      
    
#    errorCount = 0
#    
#    for item in testData:
#        prediction = dist.knn(9000,7,trainData,item)
#        
#        if not(prediction.__eq__(item[0][0])):
#            errorCount = errorCount + 1
#            
#        # end if not prediction.__eq__
#    # end for item in data
        

if __name__ == "__main__":
    main()