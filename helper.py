#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np


# In[109]:


###Activation Functions


# In[110]:


def sigmoid (z):
    ans = 1/(1+ np.exp(-1*z))
    return ans


# In[111]:


def d_sigmoid(z):
    ans = sigmoid(-z)/((1+sigmoid(-z))**2)
    return ans


# In[112]:


def ReLU(z):
    '''Leaky ReLU'''
    def func(a):
        if a<0:
            return a*0.01
        else: 
            return a
    ans = np.vectorize(func)(z)
    return ans


# In[113]:


def d_ReLU(z):
    '''Leaky ReLU derivative'''
    z[z>=0] = 1
    z[z<0] = 0.01
    return z


# In[114]:


def tanh(z):
    ans = np.tanh(z)
    return ans


# In[115]:


def d_tanh(z):
    ans = 1 - tanh(z)**2
    return ans


# In[116]:


def softmax(y):
    denom = np.sum(np.exp(y),axis = 0)
    num = np.exp(y)
    ans = num/ denom
    return ans

def d_softmax(z , y):
    return (softmax(z)-y)


# In[135]:


def initialize_param(shape):
    '''Initializes Weight W and bias b for a connection'''
    x,y = shape
    W = np.random.randn(x,y)*0.01
    b = np.zeros((x,1))
    return W,b
    


# In[136]:
def grad_descent(Z,dZ, learning_rate):
    '''Updates parameters of Z by simple gradient descent'''
    Z = Z - learning_rate*dZ
    return Z





