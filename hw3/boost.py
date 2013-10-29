# -*- coding: utf-8 -*-
"""
Contains definitions for boosting

"""

import numpy as np

class BoostModel(object):
    # either a dict of perceptrons or stumps paired with alpha values
        
    model = list()
    model_type = ''
    stumps = None
    inverse_stumps = None
    
    def __init__(self,model_type):
        try:
            assert((model_type == 'stump' or model_type == 'perceptron' or model_type == 'both'))
            self.model_type = model_type
        except AssertionError:
            print "Assertion Error. Model type must be stump, perceptron, or both. Defaulting to stump."
            self.model_type = 'stump'
            
    
    # trains a model using adaptive boosting
       
    def train(self,x_train,y_train,x_validate,y_validate):
        if self.model_type == 'stump' or self.model_type == 'both':
            self.build_stumps(x_train,y_train)
        
        terminate = False
        H = np.zeros((np.shape(y_validate)[0]),dtype=np.float)
        w = np.ones((np.shape(y_train)[0],),dtype=np.float)
        w = w / sum(w)
        # smoothing constant to ensure we don't divide by 0
        epsilon = 1e-6
        # validation accuracy so we know when to stop
        validate = list()        
        count = 0        
        
        while terminate is False:
            h,acc,best_stump = self.choose_stump(w,y_train)
            alpha = .5 * np.log((acc + epsilon)/((1 - acc) + epsilon))
            best_stump['alpha'] = alpha
            self.model.append(best_stump)
            # weight update + normalize            
            w = w * (np.e**-(alpha*y_train*h))
            w = w / sum(w)
            # validate
            positive = np.where(x_validate[best_stump['ind'],:] == best_stump['pos'])            
            negative = np.where(x_validate[best_stump['ind'],:] == best_stump['neg'])            
            H[positive] = H[positive] + alpha
            H[negative] = H[negative] - alpha
            perf = (np.sign(H) == y_validate).astype(np.float)
            print "%.5f" % (sum(perf)/len(perf))
            print "Best index: %d Democrat = %d Acc of stump: %.5f" % (best_stump['ind'],best_stump['pos'],acc)
            
            validate.append(np.int(np.float(sum(perf)/np.float(len(perf))) * 100))
            if len(validate) > 1:
                if validate[-1] <= validate[-2]:
                    count = count + 1
                else:
                    count = 0
            
            if count == 10:
                terminate = True
                self.model = self.model[0:-10]
        
        
            
    # end train()
   
    # trains perceptrons
    
    def perceptron(self,x,y):
        print ""    
    
   
    def choose_stump(self,w,y):
        acc_max = 0
        best_stump = dict()
        
        for i in range(np.shape(self.stumps)[0]):            
            # consider the stump and it's opposite            
            perf = y * self.stumps[i,:]
            neg_perf = y * (self.stumps[i,:] * -1)
            perf[np.where(perf == -1)] = 0
            neg_perf[np.where(perf == -1)] = 0
            acc_neg_stump = np.dot(neg_perf,w)
            acc_stump = np.dot(perf,w)
            
            if acc_neg_stump > acc_stump and acc_neg_stump > acc_max:
                h = self.stumps[i,:] * -1
                acc_max = acc_neg_stump
                best_stump['ind'] = i
                best_stump['pos'] = 0
                best_stump['neg'] = 1
            elif acc_stump > acc_neg_stump and acc_stump > acc_max:
                h = self.stumps[i,:]
                acc_max = acc_stump
                best_stump['ind'] = i
                best_stump['pos'] = 1
                best_stump['neg'] = 0
        
        return (h,acc_max,best_stump)
                
    # considers all attributes (y/n) and returns the classification for each
    
    def build_stumps(self,x,y):
        self.stumps = np.zeros((np.shape(x)[0],np.shape(x)[1]),dtype=np.int)
        # split on each value, 1 = democract, = 0 republican
        # 1 = classified as democrat, -1 = classified as republican
        # multiplying by -1 gets the opposite classification
        for j in range(np.shape(x)[1]):
            truthiness = x[:,j]            
            truthiness[np.where(truthiness == 0)] = -1
            self.stumps[:,j] = truthiness
        
        
    # end build_stumps()

    
    # evalutes a particular perceptron (defined as a lambda function) for training
    # data             
    
    def eval_perceptron(self,perceptron,x,y):
        print ""
        
    # end eval_perceptron()
    
    
    # evalutes novel examples
       
    def evaluate(self,x,y):
        H = np.zeros((np.shape(y)[0]),dtype=np.float)
        for stump in self.model:
            positive = np.where(x[stump['ind'],:] == stump['pos'])
            negative = np.where(x[stump['ind'],:] == stump['neg'])            
            H[positive] = H[positive] + stump['alpha']
            H[negative] = H[negative] - stump['alpha']
        
        perf = (np.sign(H) == y).astype(np.float)        
        return float(sum(perf)/len(perf))   
        
    # end evaluate()
        