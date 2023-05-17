import os, sys, csv, cv2, random
import numpy as np
from kNN import NearestNeighborClasiffier
from Linear import LinearClasiffier
from load_data import load_data, get_classes_num
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.optimizers import Adam
from keras import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,accuracy_score
###
load_data.N_OF_CLASSES = 36 # 36
##
path = os.path.dirname(os.path.abspath(__file__))
klasa = ["Zakazu", "Ostrzegawczy", "Nakazu"]
opis = ["Ograniczenie prędkości do 20 km/h","Ograniczenie prędkości do 30 km/h","Ograniczenie prędkości do 50 km/h",
        "Ograniczenie prędkości do 60 km/h","Ograniczenie prędkości do 70 km/h","Ograniczenie prędkości do 80 km/h",
        "Ograniczenie prędkości do 100 km/h","Ograniczenie prędkości do 120 km/h", "Zakaz wyprzedzania", "Zakaz wyprzedzania przez samochody ciężarowe",
        "Zakaz ruchu w obu kierunkach", "zakaz wjazdu samochodów ciężarowych","zakaz wjazdu", "skrzyżowanie z drogą podporządkowaną", "inne niebezpieczeństwo",
        "niebezpieczny zakręt w lewo","niebezpieczny zakręt w prawo","niebezpieczne zakręty - pierwszy w lewo", "nierówna droga", "śliska jezdnia",
        "zwężenie jezdni - prawostronne", "roboty na drodze","sygnały świetlne","przejście dla pieszych","dzieci","rowerzyści!!!", "oszronienie jezdni","zwierzęta dzikie",
        "nakaz jazdy w prawo za znakiem", "nakaz jazdy w lewo za znakiem", "nakaz jazdy prosto", "nakaz jazdy prosto lub w prawo", "nakaz jazdy prosto lub w lewo",
        "nakaz jazdy z prawej strony znaku", "nakaz jazdy z lewej strony znaku", "ruch okrężny"]

def nadrzedna(klasa):
    if klasa >= 0 and klasa <= 12:
        return 0
    if klasa >= 13 and klasa <= 27:
        return 1
    if klasa >=28 and klasa <= 35:
        return 2

def load_single(file):   
    piksele=[]
    image = cv2.imread(os.path.join(path,file))
    newimage = cv2.resize(image,(32, 32))
    #piksele.append(np.array(newimage).flatten())
    piksele.append(np.array(newimage))
    return np.array(piksele)

#model = LinearClasiffier(klasy=get_classes_num(), metoda="localsearch",iters=10000,step=0.00001)
#model.wczytaj_model("./Modele/model_36_100000")
model = load_model('Modele/nsiec36')

print("start")
while True:
    inp = str(input())
    X_test = load_single("zdjTestowe\\"+inp+".png")
    #prd_idx = int(model.predict(X_test))
    prd_idx = int(np.argmax(model.predict(X_test), axis=-1))
    print(f"Znak {klasa[nadrzedna(prd_idx)]}\nOpis: {opis[prd_idx]}")
    #print(prd_idx)