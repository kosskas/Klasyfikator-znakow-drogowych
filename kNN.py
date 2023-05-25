import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class NearestNeighborClasiffier:
    def __init__(self,klasy,k, norma):
        self.__CLASS_NUM__ = klasy
        self.p = norma
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def evaluate(self, X, y):
        y_predict = self.predict(X)
        avg = round(np.mean(y_predict == y),4)
        print(f"{avg}")

        confmat = confusion_matrix(y,y_predict)
        pred = confmat.diagonal()/confmat.sum(axis=1)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                print(confmat[i][j], end=" ")
            print()
        for i in [i for i in range(self.__CLASS_NUM__)]:
            print("Klasa [{0}] = {1:.2f}".format(i,pred[i]))
        return avg, confmat

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype = self.y_train.dtype)
        for i in range(X.shape[0]):
            dist = self.norma(X, i)
            neighbor_idx = np.argsort(dist)[:self.k] # wez indexy k najblizszych sasiadow
            zlicz = np.bincount(neighbor_idx) # zlicz wystąpienia
            idx_min = np.argmax(zlicz) # wybierz najczęstszy index
            y_pred[i] = self.y_train[idx_min] # klasą obj testowego jest klasa najb. sąsiada
        return y_pred

    def norma(self, X, i):
        return np.power(np.sum(np.power(np.abs(self.X_train - X[i,:]),self.p),axis = 1),1/self.p)