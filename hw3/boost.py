# -*- coding: utf-8 -*-
"""
Contains definitions for boosting

"""

import numpy as np
import random

class BoostModel(object):
    # either a dict of perceptrons or stumps paired with alpha values
        
    model = list()
    model_type = ''
    perceptrons = None
    perceptron_stumps = None
    stumps = None
    inverse_stumps = None
    
    def __init__(self,model_type):
        try:
            assert((model_type == 'stump' or model_type == 'perceptron' or model_type == 'both'))
            self.model_type = model_type
        except AssertionError:
            print "Assertion Error. Model type must be stump, perceptron, or both. Defaulting to stump."
            self.model_type = 'stump'
            
    
    # trains a model using adaptive boosting w/ perceptrons

    def train_perceptron(self,x_train,y_train,x_validate,y_validate):
        self.build_perceptrons(x_train,y_train,100,.2)
        terminate = False
        H = np.zeros((np.shape(y_validate)[0],),dtype=np.float)
        w = np.ones((np.shape(y_train)[0],),dtype=np.float)
        w = w / sum(w)
        # smoothing constant to ensure we don't divide by 0
        epsilon = 1e-6
        # validation accuracy so we know when to stop
        validate = list()        
        count = 0   
        
        while terminate is False:
            h,acc,best_stump = self.choose_perceptron(w,y_train)
            alpha = .5 * np.log((acc + epsilon)/((1 - acc) + epsilon))
            best_stump['alpha'] = alpha
            self.model.append(best_stump)
            # weight update + normalize            
            w = w * (np.e**-(alpha*y_train*h))
            w = w / sum(w)
            h,throwaway = self.eval_perceptron(x_validate,y_validate,self.perceptrons[best_stump['ind']],True)
            H = H + h*alpha
            
            perf = (np.sign(H) == y_validate).astype(np.float)
            
            validate.append(np.int(np.float(sum(perf)/np.float(len(perf))) * 100))
            if len(validate) > 1:
                if validate[-1] <= validate[-2]:
                    count = count + 1
                else:
                    count = 0
            
            if count == 10:
                terminate = True
                self.model = self.model[0:-10]           
            
            
    # trains a model using adaptive boosting w/ stumps
       
    def train_stump(self,x_train,y_train,x_validate,y_validate):
        self.build_stumps(x_train,y_train)
        
        terminate = False
        H = np.zeros((np.shape(y_validate)[0],),dtype=np.float)
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
   
    # trains a 'pile' of perceptrons
    # n is the number of perceptrons to train
    # ata is the learning rate    
    
    def build_perceptrons(self,x,y,n,eta):
        self.perceptron_stumps = np.zeros((n,np.shape(x)[1]),dtype=np.int)
        self.perceptrons = list()
        
        for xPerceptron in range(n):
            w = np.zeros((16,),dtype=np.float)
            w[:] = [random.uniform(-2,2) for i in range(16)]
            b = -1
            num_attempts = 0
            # randomly initialized weights could have accuracy > 50%            
            h,acc = self.eval_perceptron(x,y,w,True)
            while acc < .5 and num_attempts < 10:
                for item in np.where(h != y):
                    w = w + eta*((y[0]-h[0])*x[:,item[0]])
                num_attempts = num_attempts + 1
                h,acc = self.eval_perceptron(x,y,w,True)
                
            self.perceptron_stumps[xPerceptron,:] = h
            self.perceptrons.append(w)            
 
    def choose_perceptron(self,w,y):
        acc_max = 0
        best_stump = dict()
        
        for i in range(np.shape(self.perceptron_stumps)[0]):            
            # consider the stump and it's opposite            
            perf = y * self.perceptron_stumps[i,:]
            perf[np.where(perf == -1)] = 0
            acc_stump = np.dot(perf,w)

            if acc_stump > acc_max:
                h = self.perceptron_stumps[i,:]
                acc_max = acc_stump
                best_stump['ind'] = i
        
        return (h,acc_max,best_stump)
  
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
    
    def eval_perceptron(self,x,y,w,return_h):
        perceptron = lambda w,b,x: np.sign(np.dot(w,x)+b)
        h = np.zeros((np.shape(x)[1],),dtype=np.int)

        for j in range(np.shape(x)[1]):
            h[j] = np.sign(perceptron(w,-1,x[:,j]))

        result = (h == y).astype(int)
        
        if return_h:
            return (h,float(sum(result) / len(result)))
        else:
            return float(sum(result) / len(result))
    # end eval_perceptron()
    
    
    # evalutes novel examples
       
    def evaluate(self,x,y):
        if self.model_type == 'stump':        
            H = np.zeros((np.shape(y)[0]),dtype=np.float)
            for learner in self.model:
                positive = np.where(x[learner['ind'],:] == learner['pos'])
                negative = np.where(x[learner['ind'],:] == learner['neg'])            
                H[positive] = H[positive] + learner['alpha']
                H[negative] = H[negative] - learner['alpha']
            
            perf = (np.sign(H) == y).astype(np.float)  
        if self.model_type == 'perceptron':
            H = np.zeros((np.shape(y)[0]),dtype=np.float)
            for learner in self.model:
                h,throwaway = self.eval_perceptron(x,y,self.perceptrons[learner['ind']],True)
                H = H + h*learner['alpha']
            perf = (np.sign(H) == y).astype(np.float)
        return float(sum(perf)/len(perf))   
        
    # end evaluate()
        