import numpy as np
import pandas as pd
from PIL import Image
import os, cv2, random, time
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class NearestNeighborClasiffier:
    def __init__(self,klasy, norma = 1, k = None):
        self.__CLASS_NUM__ = klasy
        self.p = norma
        if k == None:
            self.k = 1
        else:
            self.k = k

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

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

    def predict(self, X, k = None):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            dist = self.norma(X, i)
            min_idx = np.argsort(dist)[:k] # wez indexy k najblizszych sasiadow
            zlicz = np.bincount(min_idx) # zlicz wystąpienia
            min_index = np.argmax(zlicz) # wybierz najczęstrzy index
            Ypred[i] = self.ytr[min_index] # klasą obj testowego jest klasa najb. sąsiada
        return Ypred

    def norma1(self, X, i):
        return np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)

    def norma2(self, X, i):
        return np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))

    def norma(self, X, i):
        return np.power(np.sum(np.power(np.abs(self.Xtr - X[i,:]),self.p),axis = 1),1/self.p)