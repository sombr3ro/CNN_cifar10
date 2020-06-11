#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
from helper import *
from layers import *


# In[86]:


class models:
    
    def __init__(self, input_shape, output_shape, learning_rate = 0.0075, optimizer ='grad_descent', cost_fn = 'cross_entropy', *args):
        
        #storing the attributes
        self.layers = args
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.cost_fn = cost_fn
        self.num_layers = len(args)
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        #Connecting all the layers of the neural model
        X = np.zeros(input_shape)
        self.layers[0].connector(X, self.optimizer)
        for i in range(1, self.num_layers):
            self.layers[i].connector(self.layers[i-1], self.optimizer)
        pass
    
    
    def forward_prop(self,X):
        self.layers[0].forward_prop(X)
        for i in range(1, self.num_layers):
            self.layers[i].forward_prop( self.layers[i-1])
        self.Y_predict = softmax(self.layers[-1].get_A())
        pass
    
    def backward_prop(self,X, Y):
        dA_last = self.Y_predict - Y
        self.layers[-1].set_dA(dA_last)
        for i in range( self.num_layers-1 , 0, -1):
            self.layers[i].back_prop(self.layers[i-1])
            self.layers[i].optimize(self.learning_rate)
        self.layers[0].back_prop(X , first=True)
        self.layers[0].optimize(self.learning_rate)
        pass
    
    def predict(self, X):
        self.forward_prop(X)  
        return np.squeeze(self.Y_predict.argmax(axis=0))
    
    def cost(self, Y):
        m = Y.shape[1]
        if (self.cost_fn=='cross_entropy'):
            cost = 1/m*(-1*Y*np.log(self.Y_predict)).sum()
        return cost
    
    def train(self,X,Y, num_epochs, batch_size = 0):
        if (batch_size==0):
            batch_size = Y.shape[1]
        batches = Y.shape[1]//batch_size
        for i in range(num_epochs):
            self.forward_prop(X)
            self.backward_prop(X,Y)
            cost = self.cost(Y)
            print ("Epoch {} over, cost: {}".format(i,cost))
        print("Model Trained")
        pass
            
    

