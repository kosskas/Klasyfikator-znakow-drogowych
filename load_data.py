import os, sys, csv, random, cv2
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))

global N_OF_CLASSES
def get_classes_num():
    return load_data.N_OF_CLASSES

def load_images(file):
    plik = pd.read_csv(file, delimiter=",")
    dane = [tuple(linia) for linia in plik.values]
    random.shuffle(dane) #mieszanie
    zrodlo = [linia[0] for linia in dane]
    if load_data.N_OF_CLASSES == 3:
        klasa = [linia[2] for linia in dane]
    else:
        klasa = [linia[1] for linia in dane]     
    piksele=[]
    for sciezka in zrodlo:
        image = cv2.imread(os.path.join(path,sciezka))
        piksele.append(np.array(image).flatten())
    zdj = np.array(piksele)
    klasa = np.array(klasa)
    return zdj, klasa

def load_data():
    X_train, Y_train = load_images("train.csv")
    X_test, Y_test = load_images("test.csv")
    #X_test, Y_test = load_images("jeden.csv")
    X_train = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # zamiast listy macierzy, lista wektorów jednowymiarowych
    X_test = X_test.reshape(X_test.shape[0], 32 * 32 * 3) 
    return (X_train, Y_train), (X_test, Y_test)

