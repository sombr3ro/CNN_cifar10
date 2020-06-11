#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helper import *
import numpy as np


# In[2]:


class dense:
    '''Fully connected dense layer'''
    
    def __init__(self, size, activation = 'ReLU'):
        self.n = size               #number of units in layer
        
        # Specifying the activation function and its derivative
        if (activation == 'ReLU'):
            self.act = ReLU
            self.d_act = d_ReLU
        elif (activation == 'sigmoid'):
            self.act = sigmoid
            self.d_act = d_sigmoid
        elif (activation == 'tanh'):
            self.act = tanh
            self.d_act = d_tanh
        else :
            self.act = np.array
            self.d_act = np.array
        
        pass
    
    
    def connector(self, prev_layer,  optimizer = 'grad_descent'):
        '''Specifies connection to the previous layer by creating W,b, dW, db and specifies the optimizer function'''
        if ( type(prev_layer)==np.ndarray):
            prev_layer_size = prev_layer.shape[0]
        else:
            prev_layer_size = prev_layer.get_layer_size()
        
        shape = (self.n , prev_layer_size)
        self.W , self.b = initialize_param(shape)
        self.dW = np.zeros(shape)
        self.db = np.zeros((prev_layer_size, 1))
        
        # Specifying the optimizer to be used for gradient descent 
        if (optimizer == 'grad_descent'):
            self.opt = grad_descent
        pass
    
    
    def forward_prop (self, prev_layer):
        if (type(prev_layer)==np.ndarray):
            A_prev = prev_layer
        else:
            A_prev = prev_layer.get_A()
        self.Z = np.dot(self.W , A_prev) + self.b
        self.A = self.act(self.Z)
        
        pass
    
    
    def back_prop(self, prev_layer, first = False):
        if (type(prev_layer)==np.ndarray):
            A_prev = prev_layer
        else:
            A_prev = prev_layer.get_A()
        m = A_prev.shape[1]
        
        self.dZ = self.dA * self.d_act(self.Z)
        self.dW = 1/m* (np.dot(self.dZ , A_prev.T))
        self.db = 1/m* np.sum (self.dZ , axis =1, keepdims=True)
        
        if (first==True):
            return
        
        dA_prev = np.dot(self.W.T , self.dZ)
        prev_layer.set_dA(dA_prev)
        
        pass
    
    
    def set_dA(self, dA):
        self.dA = dA
        pass
    
    
    def get_A(self):
        return self.A
    
    
    def get_layer_size(self):
        return self.n
    
    
    def optimize(self, learning_rate):
        self.W = self.opt (self.W, self.dW, learning_rate)
        self.b = self.opt (self.b, self.db, learning_rate)
        pass
    
    def get_params(self):
        return self.W, self.b
    
    def set_params(W,b):
        self.W = W
        self.b = b
        pass
    
        


# In[ ]:




