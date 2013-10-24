# -*- coding: utf-8 -*-
"""
Contains definitions for boosting

"""

class Model(object):
    # either a dict of perceptrons or stumps paired with alpha values
    feats = dict()
    model_type = ''
    stumps = None
    
    def __init__(self,model_type):
        try:
            assert((model_type == 'stump' or model_type == 'perceptron' or model_type == 'both'))
            self.model_type = model_type
        except AssertionError:
            print "Assertion Error. Model type must be stump, perceptron, or both. Defaulting to stump."
            self.model_type = 'stump'
            
    @classmethod
    """ 
    trains a model using adaptive boosting
    """    
    def train(self,x,y):
        if self.model_type == 'stump' or self.model_type == 'both':
            self.buildStumps(x,y)
            
    # end train()
    @classmethod
    """
    trains perceptrons
    """
    def perceptron(self,x,y):
        
    
    @classmethod
    """
    considers all attributes (y/n) and returns the classification for each
    """
    def build_stumps(self,x,y):
        
    # end build_stumps()
    @classmethod
    """
    evalutes a particular perceptron (defined as a lambda function) for training
    data             
    """ 
    def eval_perceptron(self,perceptron,x,y):
       
    # end eval_perceptron()
    @classmethod
    """
    evalutes novel examples
    """    
    def evaluate(self,x,y):
        
    # end evaluate()
        