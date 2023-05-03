import numpy as np
import pandas as pd
from PIL import Image
import os, cv2, random, time
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from load_data import get_classes_num

class LinearClasiffier: 
    def __init__(self,klasy, metoda = None, iters = None, step = None):
        self.__CLASS_NUM__ = klasy
        metody = {"random": self.random_search, "localsearch": self.local_search}
        if metoda == None:
            self.metoda_tr = metody["localsearch"]
        else:
            self.metoda_tr = metody[metoda]
        if iters == None:
            self.iters = 1000
        else:
            self.iters = iters
        if step == None:
            self.step = 0.0001
        else:
            self.step = step

    def train(self, Xtrain, Ytrain):
        Xtrain = np.concatenate((Xtrain, np.ones((Xtrain.shape[0], 1))), axis=1).T #dodanie na koniec jedynek i transpozycja
        self.Xtr = Xtrain
        self.ytr = Ytrain
        self.metoda_tr(Xtrain, Ytrain)
        
    def evaluate(self, X, y):
        Y_predict = self.predict(X)
        avg = round(np.mean(Y_predict == y),2)
        print(f"{avg}")
        confmat = confusion_matrix(y,Y_predict)
        pred = confmat.diagonal()/confmat.sum(axis=1)
        print(confmat)
        for i in [i for i in range(self.__CLASS_NUM__)]:
            print("Klasa [{0}] = {1:.2f}".format(i,pred[i]))
        return avg

    def predict(self, Xtest):
        Xtest = np.concatenate((Xtest, np.ones((Xtest.shape[0], 1))), axis=1).T #dodanie na koniec jedynek i transpozycja
        scores = self.W.dot(Xtest) 
        Ypred = np.argmax(scores, axis= 0) # index pod największym wynikiem
        return Ypred


    def local_search(self, Xtrain, Ytrain):
        self.W = np.random.randn(self.__CLASS_NUM__, 3073) * self.step
        bestloss = float("inf")
        self.licz = []
        self.strata = []
        for i in range(self.iters):
            step_size = self.step
            Wtry = self.W + np.random.randn(self.__CLASS_NUM__, 3073) * step_size
            loss = L(Xtrain, Ytrain, Wtry)
            if loss < bestloss:
                self.W = Wtry
                bestloss = loss            
            self.licz.append(i)
            self.strata.append(loss)
            if i % 100 == 0:
                print('iteracja %d the loss was %f, best %f' % (i, loss, bestloss))

    def random_search(self, Xtrain, Ytrain):
        #raz lepiej raz gorzej
        bestloss = float("inf")
        for i in range(self.iters):        
            W_iter = np.random.randn(self.__CLASS_NUM__, 3073) * self.step
            loss = L(Xtrain, Ytrain, W_iter) 
            if loss < bestloss:
                bestloss = loss
                self.W = W_iter
            if i % 100 == 0:
                print('iteracja %d the loss was %f, best %f' % (i, loss, bestloss))

    def get_loss(self):
        return self.licz, self.strata

    def get_wagi(self):
        return self.W

    def gradient(self, W):
        raise "Nie ma gradientu"

    def zapisz_model(self,nazwa):
        with open(f'model_{nazwa}.npy', 'wb') as f:
            np.save(f, self.W)

    def wczytaj_model(self,nazwa):
        with open(f'{nazwa}.npy', 'rb') as f:
            self.W = np.load(f)

def L(X, y, W):
    scores = W.dot(X)
    margins = np.maximum(0, scores - scores[y, np.arange(X.shape[1])] + 1)
    margins[y, np.arange(X.shape[1])] = 0
    loss = np.sum(margins) / X.shape[1]
    return loss


#########################
#def L_i(x, y, W):
#    """
#    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
#    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
#    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
#    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
#    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
#    """
#    delta = 1.0 # see notes about delta later in this section
#    scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
#    correct_class_score = scores[y]
#    D = W.shape[0] # number of classes, e.g. 10
#    loss_i = 0.0
#    for j in range(D): # iterate over all wrong classes
#        if j == y:
#        # skip for the true class to only loop over incorrect classes
#            continue
#        # accumulate loss for the i-th example
#        loss_i += max(0, scores[j] - correct_class_score + delta)
#    return loss_i

#def L_i_vectorized(x, y, W):
#    """
#    A faster half-vectorized implementation. half-vectorized
#    refers to the fact that for a single example the implementation contains
#    no for loops, but there is still one loop over the examples (outside this function)
#    """
#    delta = 1.0
#    scores = W.dot(x)
#    # compute the margins for all classes in one vector operation
#    margins = np.maximum(0, scores - scores[y] + delta)
#    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
#    # to ignore the y-th position and only consider margin on max wrong class
#    margins[y] = 0
#    loss_i = np.sum(margins)
#    return loss_i

#def foll_gradient(self, Xtrain, Ytrain):
#        """ cos z gradientem, nie wiem co to jest i długo liczy
#        """
#        self.W = np.random.rand(3, 3073) * self.step
#        loss_original = self.loss_fun(self.W) # the original loss
#        print('original loss: %f' % (loss_original, ))
#        # lets see the effect of multiple step sizes
#        bestloss = float("inf")
#        for i in range(self.iters):
#            W_new = self.W - step_size * df # new position in the weight space
#            loss_new = self.loss_fun(W_new)
#            print( 'for step size %f new loss: %f' % (step_size, loss_new))