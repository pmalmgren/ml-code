import random
import numpy as np
from scipy.spatial.distance import cdist

# n number of samples, k number of neighbors
# x training set, y new example

def knn(n,k,x,y):
    # define custom data type to hold nearest neighbors
    dtNN = np.dtype([('data',np.double, 1),('labels',np.str_, 1)])    
    neighbors = np.zeros((n,1),dtNN)
    
    # get our sample
    sampleData = x[random.sample(range(15000),n)]
    
    # compute distance from query to all data points    
    dm = cdist(y['data'], np.squeeze(sampleData['data']), lambda u, v: np.sqrt(((u-v)**2).sum()))
    
    neighbors['data'] = dm.transpose()
    neighbors['labels'] = sampleData['labels']
    
    # sort data, return labels of k nearest neighbors
    knn = np.sort(neighbors,0)[1:k]['labels'].tolist()    
    
    maxCount = 0
    classifiedLabel = ''
    
    for item in knn:
        # count occurences of each item
        if knn.count(item) > maxCount:
            maxCount = knn.count(item)
            classifiedLabel = item[0]
    # end for item in knn
            
    return classifiedLabel
    
# end knn()