# -*- coding: utf-8 -*-
"""
main.py

reads in data, trains classifier using AdaBoost, then outputs a number of 
relevant statistics.

"""

import numpy as np
import random
import boost

def vectorize(data):
    vectorizedData = np.zeros((16,len(data)),dtype=np.float)    
    labels = np.zeros((len(data),),dtype=np.int)    
    
    for i in range(len(data)):
        example = data[i]
        example = example.split(',')
        if example[0] == 'republican':
            labels[i] = -1
        else:
            labels[i] = 1
            
        vector = np.zeros((16,),dtype=np.float)
        
        for j in range(16):
            if example[j+1] == 'y':
                vector[j] = 1
            elif example[j+1] == 'n':
                vector[j] = 0
            else:
                vector[j] = -1
        
        print vector
        vectorizedData[:,i] = vector
        print vectorizedData[:,i]
        
    return labels,vectorizedData
    

def main():
    with open('house-votes-84.data','r') as data_file:
        raw_data = data_file.read()
    raw_data = raw_data.strip()
    raw_data = raw_data.split('\n')
    
    labels,vectorData = vectorize(raw_data)
    randomInds = random.sample(range(len(raw_data)),len(raw_data))
    fourth = int(len(raw_data)/4)
    x_test = vectorData[:,randomInds[0:fourth]]
    y_test = labels[randomInds[0:fourth]]
    x_validate = vectorData[:,randomInds[fourth:fourth*2]]
    y_validate = labels[randomInds[fourth:fourth*2]]
    x_train = vectorData[:,randomInds[fourth*2:]]
    y_train = labels[randomInds[fourth*2:]]
    
    # train a model with a stump, perceptron, or both.
    myclassifier = boost.BoostModel('stump')
    myclassifier.train(x_train,y_train,x_validate,y_validate)
    print myclassifier.evaluate(x_test,y_test)    
    
if __name__ == '__main__':
    main()