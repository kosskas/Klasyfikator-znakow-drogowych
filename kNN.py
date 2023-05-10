import numpy as np
import pandas as pd
from PIL import Image
import os, cv2, random, time
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class NearestNeighborClasiffier:
    def __init__(self,klasy,k, norma):
        self.__CLASS_NUM__ = klasy
        self.p = norma
        self.k = k

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def evaluate(self, X, y):
        Y_predict = self.predict(X)
        avg = round(np.mean(Y_predict == y),4)
        print(f"{avg}")
        confmat = confusion_matrix(y,Y_predict)
        pred = confmat.diagonal()/confmat.sum(axis=1)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                print(confmat[i][j], end=" ")
            print()
        for i in [i for i in range(self.__CLASS_NUM__)]:
            print("Klasa [{0}] = {1:.2f}".format(i,pred[i]))
        return avg, confmat

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            dist = self.norma(X, i)
            min_idx = np.argsort(dist)[:self.k] # wez indexy k najblizszych sasiadow
            zlicz = np.bincount(min_idx) # zlicz wystąpienia
            min_index = np.argmax(zlicz) # wybierz najczęstrzy index
            Ypred[i] = self.ytr[min_index] # klasą obj testowego jest klasa najb. sąsiada
        return Ypred

    def L1(self, X, i):
        return np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)

    def norma2(self, X, i):
        return np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))

    def norma(self, X, i):
        return np.power(np.sum(np.power(np.abs(self.Xtr - X[i,:]),self.p),axis = 1),1/self.p)

##KNN###
##najlepsze k crosswalidacja
#valid = []
#Xtr, Ytr = train_data
#D = [Xtr[i:i+4000] for i in range(0, 5)]
#T = [Ytr[i:i+4000] for i in range(0,5)]
#for k in [1, 3, 5, 10, 20, 50, 100]:
#    #print(k)
#    for i in range(0, 5):
#        #print([i%5 for i in range(i, i+4)], (i+4)%5)
#        XTrData = np.concatenate([D[i%5] for i in range(i, i+4)])
#        YTrVal = np.concatenate([T[i%5] for i in range(i, i+4)])
#        TestData = D[(i+4)%5]
#        TestVal = T[(i+4)%5]
#        nn = NearestNeighborClasiffier(norma = 2) 
#        nn.train(XTrData, YTrVal)
#        Yte_predict = nn.predict(TestData, k)
#        sre = np.mean(Yte_predict == TestVal)
#        print((k,[i%5 for i in range(i, i+4)],(i+4)%5, sre ))
#        valid.append( (k,[i%5 for i in range(i, i+4)],(i+4)%5, sre ) )
#print(valid)