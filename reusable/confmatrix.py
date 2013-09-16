from nltk.metrics.confusionmatrix import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import random


""" input arguments:
	name of file to write confusion matrix to
	cm is an nltk confusion matrix
	labels is a list of classes
      reference: http://stackoverflow.com/a/5824945
"""
def animateMatrix(name,cm,labels):
    conf_arr = cm._confusion
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if float(a) != 0:
                tmp_arr.append(float(j)/float(a))
            else:
                tmp_arr.append(0)
        norm_conf.append(tmp_arr)
    
    # manipulate figsize/dpi if you need a bigger or smaller matrix
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
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    plt.savefig(name, format='png',dpi=80,figsize=(20,20))
	
def main():
    # predictions and truth are just one hundred random letters
    predictions = [string.uppercase[j] for j in [random.randint(0,25) for r in xrange(100)]]
    truth = [string.uppercase[j] for j in [random.randint(0,25) for r in xrange(100)]]
    # generates confusion matrix using NLTK library
    cm = ConfusionMatrix(predictions,truth)
    # text version of confusion matrix
    print cm
    # displays/writes confusion matrix to a file
    animateMatrix('confusionmatrx.png',cm,string.uppercase)
	
if __name__ == '__main__':
	main()