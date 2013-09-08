import string
import dist
import random
from matplotlib.mlab import find
import numpy as np
from scipy.spatial.distance import cdist

# n number of samples, k number of neighbors
# x training set, y new example

def knn(k,sampleData,y):
    # define custom data type to hold nearest neighbors
    n = len(sampleData)    
    dtNN = np.dtype([('data',np.double, 1),('labels',np.str_, 1)])    
    neighbors = np.zeros((n,1),dtNN)
    
    # compute distance from query to all data points    
    dm = cdist(y['data'], np.squeeze(sampleData['data']), lambda u, v: np.sqrt(((u-v)**2).sum()))
    
    neighbors['data'] = dm.transpose()
 
    neighbors['labels'] = np.reshape(sampleData['labels'],(len(neighbors),1))

    
    # sort data, return labels of k nearest neighbors
    knn = np.sort(neighbors,0)[0:k]['labels'].tolist()    
    
    maxCount = 0
    classifiedLabel = ''
    
    if type(k) is list:    
        print "shup"


    else:    
        if k > 1:
            for item in knn:
                # count occurences of each item
                # ties are broken based on which occurs first
                if knn.count(item) > maxCount:
                    maxCount = knn.count(item)
                    classifiedLabel = item[0]
            # end for item in knn
        else:
            classifiedLabel = knn[0][0]
            
    return classifiedLabel
    
# end knn()
    
def condense(trainData):
    
    alphabet = [string.uppercase[i:i+1] for i in range(0,len(string.uppercase))]
    minSet = np.zeros((len(trainData),1),np.bool)
    
    for i in range(0,len(alphabet)):
        letterIndices = find(trainData['labels'] == alphabet[i])
        randIndex = random.randint(0,len(letterIndices)-1)
        minSet[letterIndices[randIndex]] = True;
        
    terminate = False
    iters = 0

    while terminate == False:        
        print "Restarting. Size of condensed set: %d previous iterations: %d" % (len(minSet),iters)      
        transfers = 0 
        iters = 0
        for i in range(0,len(trainData)):
            if minSet[i]:
                continue
            else:
                label = dist.knn(1,trainData[minSet],trainData[i])
                iters = iters + 1
                if label != trainData[i][0][0]:
                    print "Error. Correct label: %s Classified label: %s" % (trainData[i][0][0],label)                    
                    transfers = transfers + 1
                    minSet[i] = True
                    
               # newIndices.remove(item)
            # end if label != trainData[item][0][0]
        # end for item in validIndices
        
        if transfers == 0:
            terminate = True
        # end if
    # end while terminate == False
            
    return trainData[minSet]
    
# end condense()