import os, sys, csv, random, cv2
import numpy as np
import pandas as pd

global N_OF_CLASSES

path = os.path.dirname(os.path.abspath(__file__))

def get_classes_num():
    return load_data.N_OF_CLASSES

def load_images(file, to_net=False):
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
        if to_net:
            piksele.append(np.array(image))
        else:
            piksele.append(np.array(image).flatten())
    return np.array(piksele), np.array(klasa)

def load_data():
    X_train, y_train = load_images("train.csv")
    X_test, y_test = load_images("test.csv")
    #X_test, y_test = load_images("jeden.csv")
    X_train = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # zamiast listy macierzy, lista wektorów jednowymiarowych
    X_test = X_test.reshape(X_test.shape[0], 32 * 32 * 3) 
    return (X_train, y_train), (X_test, y_test)

def load_to_network():
    X_train, y_train = load_images("train.csv",to_net =True)
    X_test, y_test = load_images("test.csv", to_net =True)
    return (X_train, y_train), (X_test, y_test)