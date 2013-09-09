""" knn.py
    This is the pretty version of the assignment code 
    NOTE: Does not include confusion matrix code (I used NLTK/matplotlib)
    knn(k,sampleSize,query)  
"""

import numpy as np
from scipy import spatial
from scipy.spatial.distance import cdist
import random

def condense(trainData):
    sortedInds = np.argsort(cdist(np.squeeze(trainData['data']),np.squeeze(trainData['data']),'euclidean'))
    alphabet = [string.uppercase[i:i+1] for i in range(0,len(string.uppercase))]    
    minSet = []

    # build initial set with one element/class
    for i in range(0,len(alphabet)):
        letterIndices = find(trainData['labels'] == alphabet[i])
        randIndex = random.randint(0,len(letterIndices)-1)
        minSet.append(letterIndices[randIndex])
    # end for i in range
    
    # terminate when no transfers occur or set is full
    transfers = 0
    while transfers == 0 or len(minSet) == len(trainData):
        transfers = 0
        for i in range(0,len(trainData)):
            if not(i in minSet):
                prediction = trainData['labels'][min(sortedInds[i,minSet])][0]
                if prediction != trainData['labels'][i][0]:
                    minSet.append(i)
                    transfers = transfers + 1
        # end for i in range
    # end while transfers
                    
    return trainData[minSet]
# end condense()
    
# returns a list of predictions
def knn(k,trainData,testData):
    sortedInds = np.argsort(cdist(np.squeeze(testData['data']),np.squeeze(trainData['data']),'euclidean'))
    predictions = np.squeeze(trainData['labels'][sortedInds[:,0:k]]).tolist()
    
    if k > 1:
        # convert predictions to a set (with no duplicate elements)
        # return the one with the greatest count in the list
        return [max(set(predictions[i]),key=predictions[i].count) for i in range(0,len(predictions))]
    else:
        return predictions
# end knn()

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
    
    # as per assignment
    d = [100, 1000, 2000, 5000, 10000, 15000]
    k = [1, 3, 5, 7, 9]
    
    # KNN with various parameters (sample size, number of neighbors)
    for samples in d:
        sampleData = trainData[random.sample(range(15000),samples)]
        for neighbors in k:
            predictions = knn(neighbors,sampleData,testData)
            performance = (sum(np.int16(predictions == np.squeeze(testData['labels']))) / 5000.0) * 100
            print "KNN With %d sample size and %d neighbors. %.4f%% accuracy\n" %(samples,neighbors,performance)
        # end for neighbors
    # end for samples
            
    # 1-NN using KDTree
    predictions = []
    # build KDTree
    tree = spatial.KDTree(np.squeeze(trainData['data']))
    # this was slow no matter what I tried..
    for i in range(0,len(testData)):
        dist,ind = tree.query(np.squeeze(testData['data'][i]))
        predictions.append(trainData['labels'][ind][0])
    # end for i in range
    performance = (sum(np.int16(predictions == np.squeeze(testData['labels']))) / 5000.0) * 100
    print "1-NN Using KDTree. %.4f%% accuracy\n" % (performance) 

    # 1-NN using condensed training set
    condensedData = condense(trainData)
    print "Condensed data %d large\n" % (len(condensedData))
    predictions = knn(1,condensedData,testData)
    performance = (sum(np.int16(predictions == np.squeeze(testData['labels']))) / 5000.0) * 100
    print "1-NN Using KDTree. %.4f%% accuracy\n" % (performance)    
                
if __name__ == "__main__":
    main()