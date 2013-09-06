#!/usr/bin/python2

import string
import dist
import random
from matplotlib.mlab import find

def condense(trainData):
    
    alphabet = [string.uppercase[i:i+1] for i in range(0,len(string.uppercase))]
    validIndices = range(0,len(trainData))   
    minSet = []
    
    for i in range(0,len(alphabet)):
        letterIndices = find(trainData['labels'] == alphabet[i])
        randIndex = random.randint(0,len(letterIndices)-1)
        minSet.append(validIndices.pop(validIndices.index(letterIndices[randIndex])))

    terminate = False

    while terminate == False:        
        print "Restarting. Size of condensed set: %d previous iterations: %d" % (len(minSet),iters)      
        transfers = 0 
        iters = 0
        for item in validIndices:
            label = dist.knn(1,trainData[minSet],trainData[item])
            iters = iters + 1
            if label != trainData[item][0][0]:
                transfers = transfers + 1                
                minSet.append(validIndices.pop(validIndices.index(item)))
                break
            # end if label != trainData[item][0][0]
        # end for item in validIndices
        if transfers == 0:
            terminate = True
        # end if
    # end while terminate == False
            
    return trainData[minSet]
    
    
#    # initialize set as size of train data
#    condensedSet = np.zeros((len(trainData),1),dt)     
#    
#    # build initial training set (one item per class)
#    for i in range(0,len(alphabet)):
#        letterIndices = find(trainData['labels'] == alphabet[i])
#        # select random example from class label
#        randIndex = random.randint(0,len(letterIndices)-1)
#        condensedSet[i] = trainData[letterIndices[randIndex]];
#        trainData = np.delete(trainData,letterIndices[randIndex])        
#    # end for i in range(0,len(alphabet))
#    
#    currentIndex = len(alphabet);
#    
#    for i in range(0,len(trainData)):
#        label = dist.knn(currentIndex,1,condensedSet[0:currentIndex],trainData[i])
#        
#        if not(label == trainData[i][0]):
#            condensedSet[currentIndex] = trainData[i]
#            currentIndex = currentIndex + 1            
#        
#        print "Iteration %d members added %d\n" % (i,currentIndex)
#        # end if not(label.__eq__(trainData[i][0]))
#    # end for i in range(0,len(trainData))
#    
#    # return condensed set sans empty elements                
#    return condensedSet[0:currentIndex]