import numpy as np
from scipy.spatial.distance import cdist
import time
import random
from nltk.metrics.confusionmatrix import ConfusionMatrix

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
    
    # knn with various sample sizes:

    # calculate distance matrix

    dm = cdist(np.squeeze(testData['data']),np.squeeze(trainData['data']),'euclidean')
    alphabet = [alphabet[i:i+1] for i in range(0,len(alphabet))]   
    
    cmTemplate = ConfusionMatrix() 
    
    d = [100, 1000, 2000, 5000, 10000, 15000]
    k = [1,3,5,7,9]
    
        
    
#    sortedInds = np.argsort(dm)
    