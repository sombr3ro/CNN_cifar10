##This script is used to train the neural model


import cifar10_web as cif 
from model import *
from layers import *

def accuracy(Y_pred, Y):
    n = len(Y)
    correct = 0.0
    for i in range(n):
        if (Y[i]==Y_pred[i]):
            correct+=1.0
    accuracy = float(correct/n*100.0)
    print ('{} per accuracy'.format(accuracy))

if __name__=='__main__':
    X_train,Y_train, X_test, Y_test = cif.cifar10(path=None)
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
   
    deep = models(

        X_train.shape[0],
        Y_train.shape[0],
        2.0,
        'grad_descent',
        'cross_entropy',
        dense(500),
        dense(200),
        dense(50),
        dense(10)
    )
    deep.train(X_train,Y_train, 1)
    
    Y_pred = np.squeeze(deep.predict(X_train))
    Y = np.squeeze(Y_train.argmax(axis=0))
    print ("Train set:")
    accuracy(Y_pred,Y)

    Y_pred = np.squeeze(deep.predict(X_test))
    Y = np.squeeze(Y_test.argmax(axis=0))
    print ("Test set:")
    accuracy(Y_pred,Y)   

    deep.save_params("./params/model.txt")

 



    '''        dense(500),
        dense(200),
        dense(50),
        dense(10),'''