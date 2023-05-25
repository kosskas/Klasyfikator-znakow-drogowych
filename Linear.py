import numpy as np
import pandas as pd
import os, cv2, random, time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from load_data import get_classes_num

class LinearClasiffier: 
    def __init__(self,klasy,iters = None, step = None):
        self.__CLASS_NUM__ = klasy
        if iters == None:
            self.iters = 1000
        else:
            self.iters = iters
        if step == None:
            self.step = 0.0001
        else:
            self.step = step
        self.W = np.random.randn(self.__CLASS_NUM__, 3073) * self.step #inicjalizuj w losowym miejscu
        self.licz = []
        self.strata = []
        self.strataVal = []
        self.accTr = []
        self.accVal = []

    def train(self, Xtrain, Ytrain, X_test, y_test):
        Xtrain = np.concatenate((Xtrain, np.ones((Xtrain.shape[0], 1))), axis=1).T #dodanie na koniec jedynek i transpozycja (bias trick)
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1).T
        bestloss = float("inf")
        for i in range(self.iters):
            tempW = self.W + np.random.randn(self.__CLASS_NUM__, 3073) * self.step # poruszaj się po sąsiedztwie
            loss = loss_fun(Xtrain, Ytrain, tempW) # oblicz stratę
            if loss < bestloss: # sprawdz czy zmalała
                self.W = tempW 
                bestloss = loss        
            valLoss = loss_fun(X_test, y_test, tempW)
            accT = self.predIter(Xtrain, Ytrain)
            accV = self.predIter(X_test, y_test)
            self.licz.append(i)
            self.accTr.append(accT)
            self.accVal.append(accV)
            self.strata.append(loss)
            self.strataVal.append(valLoss)
            
            if i % 100 == 0:
                print('iteracja {0}, strata {1}, najlepsza {2}'.format(i, loss, bestloss))
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        avg = round(np.mean(y_pred == y_test),4)
        print(f"{avg}")
        confmat = confusion_matrix(y_test,y_pred)

        pred = confmat.diagonal()/confmat.sum(axis=1)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                print(confmat[i][j], end=" ")
            print()
        for i in [i for i in range(self.__CLASS_NUM__)]:
            print("Klasa [{0}] = {1:.2f}".format(i,pred[i]))
        return avg, confmat

    def predict(self, X_test):
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1).T #dodanie na koniec jedynek i transpozycja
        scores = self.W.dot(X_test) 
        ypred = np.argmax(scores, axis= 0) # index pod największym wynikiem
        return ypred

    def predIter(self, X_test, y_test):
        scores = self.W.dot(X_test) 
        y_pred = np.argmax(scores, axis= 0)
        return round(np.mean(y_pred == y_test),4)

    def get_loss(self):
        return self.licz, self.strata, self.strataVal

    def get_acc(self):
        return self.licz, self.accTr, self.accVal

    def get_wagi(self):
        return self.W

    def zapisz_model(self,nazwa):
        with open(f'model_{nazwa}.npy', 'wb') as f:
            np.save(f, self.W)

    def wczytaj_model(self,nazwa):
        with open(f'{nazwa}.npy', 'rb') as f:
            self.W = np.load(f)

    def train(self, Xtrain, Ytrain, X_test, y_test):
        Xtrain = np.concatenate((Xtrain, np.ones((Xtrain.shape[0], 1))), axis=1).T #dodanie na koniec jedynek i transpozycja (bias trick)
        X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1).T
        bestloss = float("inf")
        for i in range(self.iters):
            tempW = self.W + np.random.randn(self.__CLASS_NUM__, 3073) * self.step # poruszaj się po sąsiedztwie
            loss = loss_fun(Xtrain, Ytrain, tempW) # oblicz stratę
            if loss < bestloss: # sprawdz czy zmalała
                self.W = tempW 
                bestloss = loss        
            valLoss = loss_fun(X_test, y_test, tempW)
            accT = self.predIter(Xtrain, Ytrain)
            accV = self.predIter(X_test, y_test)
            self.licz.append(i)
            self.accTr.append(accT)
            self.accVal.append(accV)
            self.strata.append(loss)
            self.strataVal.append(valLoss)

def loss_fun(X, y, W):
    scores = W.dot(X) # oblicz scores dla bieżących wag
    # poprawny wynik powinien być większy niż suma niepoprawnych
    # hard margin
    margins = np.maximum(0, scores - scores[y, np.arange(X.shape[1])] + 1) # hinge loss
    margins[y, np.arange(X.shape[1])] = 0     # poprawne pomiń
    return np.sum(margins) / X.shape[1] # zwróć srednią wszystkich strat