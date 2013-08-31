# n number of samples, k number of neighbors
# x training set, y new example

import random
from scipy.spatial.distance import cdist

def knn(n,k,x,y):
    
    # get our sample
    sampleData = x[random.sample(range(15000),n)]
    
    
    
    dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))