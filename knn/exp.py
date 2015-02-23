import numpy as np
from scipy.spatial.distance import cdist
from scipy import spatial
import time
import random
import string
from nltk.metrics.confusionmatrix import ConfusionMatrix
import matplotlib.pyplot as plt

def animateMatrix(name,cm):
    conf_arr = cm._confusion
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    
    # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')
    
    width = len(conf_arr)
    height = len(conf_arr[0])
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig(name, format='png',dpi=80,figsize=(20,20))

def condense(trainData):
    sortedInds = np.argsort(cdist(np.squeeze(trainData['data']),np.squeeze(trainData['data']),'euclidean'))
    alphabet = [string.uppercase[i:i+1] for i in range(0,len(string.uppercase))]    
    minSet = []

    # build initial set with one element/class

    for i in range(0,len(alphabet)):
        letterIndices = find(trainData['labels'] == alphabet[i])
        randIndex = random.randint(0,len(letterIndices)-1)
        minSet.append(letterIndices[randIndex])
    
    # we terminate when no transfers occur or set is full
    transfers = 0
    
    while transfers == 0 or len(minSet) == len(trainData):
        transfers = 0
        for i in range(0,len(trainData)):
            if not(i in minSet):
                prediction = trainData['labels'][min(sortedInds[i,minSet])][0]
                if prediction != trainData['labels'][i][0]:
                    minSet.append(i)
                    transfers = transfers + 1
                    
    return trainData[minSet]

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

    # calculate distance between all testing data and training data
    dataFile = open('expdata.txt','w+')    
    t1 = time.time()
    dm = cdist(np.squeeze(testData['data']),np.squeeze(trainData['data']),'euclidean')
    t2 = time.time()
    dataFile.write('Distance matrix time: ' + str((t2-t1)*1000))
    
    d = [100, 1000, 2000, 5000, 10000, 15000]
    k = [1, 3, 5, 7, 9]
   
    
    for sampleSize in d:
        # get the sorted indices of a random sample size
        sampleInds = random.sample(range(15000),sampleSize)
        sampleLabels = trainData[sampleInds]
        t1 = time.time()        
        sortedInds = np.argsort(dm[:,sampleInds])
        t2 = time.time()
        sortTime = (t2-t1)*1000
        dataFile.write('Sort time for d ' + str(sampleSize) + ': ' + str(sortTime))
        for neighbors in k:
            ref = []
            test = []
            print "Calculating neighbors %d sample size %d" % (sampleSize,neighbors)
            t1 = time.time()            
            predictions = squeeze(sampleLabels['labels'][sortedInds[:,0:neighbors]]).tolist()                        
            for i in range(0,len(predictions)):
                ref.append(testData['labels'][i][0])
                test.append(max(set(predictions[i]),key=predictions[i].count))
            t2 = time.time()
            cm = ConfusionMatrix(ref,test)
            dataFile.write('Sample size: ' + str(sampleSize) + ' Neighbors: ' + str(neighbors) + '\n')
            dataFile.write('Running time: ' + str((t2-t1)*1000) + 'ms\n')
            dataFile.write('Percentage Correct: ' + str(cm._correct) + '/5000 = ' + str((cm._correct/5000.0)*100) + '%\n')            
            dataFile.write('Confusion matrix: \n')
            dataFile.write(cm.pp())
 #           animateMatrix(str('knn' + str(sampleSize) + str(neighbors) + '.png'),cm)
            
    tree = spatial.KDTree(np.squeeze(trainData['data']))
    ref = []
    test = []
    dataFile.write('KDTree 1NN\n')    
    t1 = time.time()
    for i in range(0,len(testData)):
        dist,ind = tree.query(np.squeeze(testData['data'][i]))
        ref.append(testData['labels'][i][0])
        test.append(trainData['labels'][ind][0])
    t2 = time.time()
    cm = ConfusionMatrix(ref,test)
    dataFile.write('Running time: ' + str((t2-t1)*1000) + 'ms\n')
    dataFile.write('Percentage Correct: ' + str(cm._correct) + '/5000 = ' + str((cm._correct/5000.0)*100) + '%\n')            
    dataFile.write('Confusion matrix: \n')
    dataFile.write(cm.pp())
#    animateMatrix(str('kdtree' + '.png'),cm)

    dataFile.write('Condensed Set 1NN\n')     
    t1 = time.time()
    
    condensedSet = condense(trainData)
    sortedInds = np.argsort(cdist(np.squeeze(testData['data']),np.squeeze(condensedSet['data']),'euclidean'))
    predictions = squeeze(condensedSet['labels'][sortedInds[:,0]])
    labels = squeeze(testData['labels'])   
    
    t2 = time.time()
    
    cm = ConfusionMatrix(labels.tolist(),predictions.tolist())
 #   animateMatrix(str('condensed' + '.png'),cm)    
    
    dataFile.write('Running time: ' + str((t2-t1)*1000) + 'ms\n')
    dataFile.write('Percentage Correct: ' + str(cm._correct) + '/5000 = ' + str((cm._correct/5000.0)*100) + '%\n')            
    dataFile.write('Confusion matrix: \n')
    dataFile.write(cm.pp())        
    
    dataFile.close()    
    
if __name__ == '__main__':
    main()