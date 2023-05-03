from PIL import Image
import os, sys, csv, cv2, random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kNN import NearestNeighborClasiffier
from Linear import LinearClasiffier
from load_data import load_data, get_classes_num

###
load_data.N_OF_CLASSES = 3 # 36
##
print(f"liczba klas = {get_classes_num()}")
print("wczytywanie")
train_data, test_data = load_data()
print("start")
#model = NearestNeighborClasiffier(norma = 1)
#model = LinearClasiffier(klasy=get_classes_num(), metoda="localsearch",iters=1000,step=0.0001)
##model.wczytaj_model("rr")
#model.train(*train_data)
#model.evaluate(*test_data)
#model.zapisz_model("36_1000")
#
#i, ls = model.get_loss()
#plt.plot(i, ls)
#plt.title("Wartość funkcji straty w kolejnych iteracjach")
#plt.xlabel("Iteracje")
#plt.ylabel("Funkcja straty")
#plt.show()

###############

for j in [-5, -6, -7]:
    step = 10**j
    print(f"LR = {10**j}")
    model = LinearClasiffier(klasy=get_classes_num(),metoda="localsearch",iters=10000,step=step)
    model.train(*train_data)
    i, ls = model.get_loss()
    model.evaluate(*test_data)
    plt.plot(i, ls, label = f"LR = {10**j}")
plt.legend(loc="upper right")
plt.title("Wartość funkcji straty w kolejnych iteracjach")
plt.xlabel("Iteracje")
plt.ylabel("Funkcja straty")
plt.show()





##KNN###
##najlepsze k crosswalidacja
#valid = []
#Xtr, Ytr = train_data
#D = [Xtr[i:i+1000] for i in range(0, 5)]
#T = [Ytr[i:i+1000] for i in range(0,5)]
#for k in [1, 2, 3, 5, 10, 20, 50, 100]:
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