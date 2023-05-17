import numpy as np
import pandas as pd
from PIL import Image
import os, cv2, random, time
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,accuracy_score
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
        avg = round(np.mean(Y_predict == y),4)
        print(f"{avg}")
        confmat = confusion_matrix(y,Y_predict)

        precision = round(precision_score(y, Y_predict,average='weighted'),4)
        recall = round(recall_score(y, Y_predict,average='weighted'),4)
        f1 = round(f1_score(y, Y_predict,average='weighted'),4)

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1}")

        pred = confmat.diagonal()/confmat.sum(axis=1)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                print(confmat[i][j], end=" ")
            print()
        for i in [i for i in range(self.__CLASS_NUM__)]:
            print("Klasa [{0}] = {1:.2f}".format(i,pred[i]))
        return avg, confmat

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
            Wnew = self.W + np.random.randn(self.__CLASS_NUM__, 3073) * step_size
            loss = L(Xtrain, Ytrain, Wnew)
            if loss < bestloss:
                self.W = Wnew
                bestloss = loss            
            self.licz.append(i)
            self.strata.append(loss)
            if i % 100 == 0:
                print('iteracja {0} the loss was {1}, best {2}'.format(i, loss, bestloss))

    def random_search(self, Xtrain, Ytrain):
        #raz lepiej raz gorzej
        bestloss = float("inf")
        for i in range(self.iters):        
            Wnew = np.random.randn(self.__CLASS_NUM__, 3073) * self.step
            loss = L(Xtrain, Ytrain, Wnew) 
            if loss < bestloss:
                bestloss = loss
                self.W = Wnew
            if i % 100 == 0:
                print('iteracja {0} the loss was {1}, best {2}'.format(i, loss, bestloss))

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