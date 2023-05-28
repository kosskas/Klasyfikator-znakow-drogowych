from PIL import Image
import cv2
import os, sys, csv, random,re
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.optimizers import Adam
from keras import *
from load_data import load_data, get_classes_num,load_images
### Nie uruchamiać ruszać! ###
"""To jest  plik używany wyłącznie do konfiguracji danych """
model = load_model('Modele/4nsiec36')
model.summary()
tf.keras.utils.plot_model(model, to_file="CNN.png", show_shapes=True)


"""precision, recall, f1_score, accuracy"""
#def prf1a(confmat):
#    true_positives = np.diag(confmat)
#    false_positives = np.sum(confmat, axis=1) - true_positives
#    false_negatives = np.sum(confmat, axis=0) - true_positives
    
#    precision = round(np.mean(true_positives / (true_positives + false_positives)),4)
#    recall = round(np.mean(true_positives / (true_positives + false_negatives)),4)
#    f1_score = round(np.mean(2 * (precision * recall) / (precision + recall)),4)
#    accuracy = round(np.sum(true_positives) / np.sum(confmat),4)
    
#    return precision, recall, f1_score, accuracy

#confmat = np.loadtxt("macierz.txt",usecols= range(3))
#precision, recall, f1_score, accuracy = prf1a(confmat)
#print("Dokładność:", accuracy)
#print("Precyzja:", precision)
#print("Zwrot:", recall)
#print("Miara F1:", f1_score)

"""rw"""
#def zapisz(macierz,nazwa):
#    with open(f'{nazwa}.npy', 'wb') as f:
#        np.save(f, macierz)

#def wczytaj(macierz,nazwa):
#    with open(f'{nazwa}.npy', 'rb') as f:
#        macierz = np.load(f)

####
#load_data.N_OF_CLASSES = 36#36
##
#print(f"liczba klas = {get_classes_num()}")
#print("wczytywanie")
#X_train, y_train = load_images("train.csv")
#print("start")
#zapisz(X_train,"XTRAIN")
#zapisz(y_train,"YTRAIN")


#model = load_model('Modele/nsiec36')
#tf.keras.utils.plot_model(model, to_file="36cnn.png", show_shapes=True)

"""wykres k"""
#k =[1,3,5,7,10,20,30,50,100]

#avg3n1 =[0.81,0.8,0.79,0.7717,0.7646,0.7375,0.716,0.6754,0.6129]
#avg3n2 =[0.83,0.8183,0.8093,0.7983,0.7907,0.7522,0.7199,0.6906,0.6328]


#avg36n1=[0.22,0.21,0.2,0.1853,0.171,0.1489,0.1446,0.1282,0.1125]
#avg36n2 = [0.26,0.2393,0.2293,0.2155,0.2045,0.1758,0.177,0.1477,0.112]

#n3 = [0.81,0.83,0.61,0.7263,0.5978,0.6043,0.4478,0.567,0.5275]

#n36 = [0.22,0.26,0.11,0.1896,0.1015,0.1355,0.0858,0.1009,0.0816]

#k =[1,3,5,7,10,20,30,50,100]

#plt.title("Dokładność w zależności od k")
#plt.ylabel("Dokładność")
#plt.xlabel("k")
##plt.bar([i for i in range(1,10)], n3, label="3 klasy")
##plt.bar([i for i in range(1,10)], n36, label="36 klas")
##plt.xticks([i for i in range(1,10)])
#plt.plot(k, avg3n1,'b', label="3 klasy L1",linestyle='dashed')
#plt.plot(k, avg3n2, 'b',label="3 klasy L2")
#plt.plot(k, avg36n1,'r', label="36 klas L1",linestyle='dashed')
#plt.plot(k, avg36n2,'r', label="36 klas L2")
#plt.xticks(k)
#plt.legend(loc="upper right")

#plt.show()


# Wczytanie macierzy konfuzji
#confmat =  np.loadtxt("macierz.txt",usecols= range(3))

## Obliczenie precision, recall, F1 Score dla każdej klasy
#n_classes = confmat.shape[0]
#precision = []
#recall = []
#f1 = []
#for i in range(n_classes):
#    tp = confmat[i, i]
#    fp = np.sum(confmat[:, i]) - tp
#    fn = np.sum(confmat[i, :]) - tp
#    p = tp / (tp + fp)
#    r = tp / (tp + fn)
#    f1_class = 2 * p * r / (p + r) if p + r > 0 else 0
#    precision.append(p)
#    recall.append(r)
#    f1.append(f1_class)

# Obliczenie wagowego F1 Score
#weighted_f1 = f1_score(y_test, y_pred, average='weighted')

# Wyświetlenie wyników
#print("Precision:", precision)
#print("Recall:", recall)
#print("F1 Score:", f1)
#print("Weighted F1 Score:", weighted_f1)

"""rysowanie macierzy"""
#confmat =  np.loadtxt("macierz.txt",usecols= range(36))
#row_sums=confmat.sum(axis=1,keepdims=True)
#norm_mat=confmat/row_sums
#norm_mat*=100

#plt.matshow(norm_mat, cmap=plt.cm.Blues)
##plt.title("Klasyfikator liniowy dla {0} klas\nDokładność = {1:.2%}\n".format(get_classes_num(),avg))
#plt.title("Klasyfikator kNN dla {0} klas\nDokładność = 26.21%\n".format(36))
#plt.ylabel("Rzeczywiste klasy")
#plt.xlabel("Przewidywane klasy")
##for i in range(confmat.shape[0]):
##    for j in range(confmat.shape[1]):
##        plt.text(j, i, f"{int(confmat[i, j])}", ha="center", va="center", color="w")
#plt.colorbar(format='%.0f%%')
#plt.clim(0, 100)
#plt.show()


#np.fill_diagonal(norm_mat,0)
#plt.matshow(norm_mat, cmap=plt.cm.Greys)
#plt.title("Błędy popełnione przy klasyfikacji")
#plt.ylabel("Rzeczywiste klasy")
#plt.xlabel("Przewidywane klasy")
##for i in range(norm_mat.shape[0]):
##    for j in range(norm_mat.shape[1]):
##        if j == i:
##            plt.text(j, i, f"0", ha="center", va="center", color="w")
##        else:
##            plt.text(j, i, f"{int(confmat[i, j])}", ha="center", va="center", color="w")
#plt.colorbar(format='%.2f%%')
#plt.show()

"""wyświetlenie wszystkich klas"""
#path= os.path.dirname(os.path.abspath(__file__))

#for r in range(20):
#    train_images = []
#    train_labels = [i for i in range(36)]
#    p = Path(path+"\Train")

#    dirs = [str(f) for f in p.iterdir() if f.is_dir()]
#    dirs = sorted(dirs, key = lambda x: (int(re.sub('\D','',x)),x))
#    #print(dirs)
#    for dir in dirs:
#        #for file in os.listdir(dir):
#        file = random.choice(os.listdir(dir))
#        image = cv2.imread(os.path.join(dir,file))
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        train_images.append(image)


#    plt.figure(figsize=(10,10))
#    for i in range(36):
#        plt.subplot(6,6,i+1)
#        plt.xticks([])
#        plt.yticks([])
#        plt.grid(False)
#        plt.imshow(train_images[i]/255.0)
#        #plt.xlabel(class_names[train_labels[i]])
#        plt.xlabel(f"[{i}]")
#    plt.show()




"""Przekształcanie bazy danych"""
#dane = []
#index = 0
#for dir in dirs:
#    klasa = os.path.basename(dir)
#    index = 0
#    for file in os.listdir(dir):
#        image = cv2.imread(os.path.join(dir,file))
#        newimage = cv2.resize(image,(32, 32))
#        cv2.imwrite(os.path.join(dir,f"{klasa}_{index}.png"), newimage)
##        #os.remove(os.path.join(dir,file))
#        index +=1 


#for dir in dirs:
#    klasa = os.path.basename(dir)
#    index = 0
#    dir_name = dir.split("\\")[-1]
#    new_path = "Train\\" + dir_name
#    for file in os.listdir(dir):
#        dane.append([new_path+"\\"+file, str(klasa), str(nadrzedna(int(klasa)))])
#np.savetxt("train.csv", 
#           dane,
#           delimiter =",", 
#           fmt ='% s')

#todel = [6, 7, 12, 13, 14, 32, 41, 42]
#przekszalc = {str(i):i-5 for i in range(33, 41)}
#    #18-27 i-4}
#plik = pd.read_csv("test1.csv", delimiter=",")
#dane = [tuple(linia) for linia in plik.values]
#zrodlo = [linia[0] for linia in dane]
#klasa = [linia[1] for linia in dane]
#sklasa= [linia[2] for linia in dane]
#dane = []
#index = 0
##for i in range(len(zrodlo)):
##    if przekszalc.get(str(klasa[i])) != None:
##        klasa[i] = przekszalc[str(klasa[i])]

#for i in range(len(zrodlo)):
#    dane.append([zrodlo[i],klasa[i], nadrzedna(int(klasa[i]))])

#np.savetxt("test2.csv", 
#           dane,
#           delimiter =",", 
#           fmt ='% s')

#for file in os.listdir("D:\\PG\\4sem\\Sztuczna Inteligencja\\Projekt\\przygotujDane\\Test"):
#    image = cv2.imread(os.path.join("D:\\PG\\4sem\\Sztuczna Inteligencja\\Projekt\\przygotujDane\\Test",file))
#    newimage = cv2.resize(image,(32, 32))
#    cv2.imwrite(os.path.join("D:\\PG\\4sem\\Sztuczna Inteligencja\\Projekt\\przygotujDane\\Test",file), newimage)

#np.savetxt("test1.csv", 
#           dane,
#           delimiter =",", 
#           fmt ='% s')
#print(file)
        #dane.append([dir+"\\"+file,str(ind)])

#for dir,ind in dirs.items():
#    klasa = ind
#    for file in os.listdir(path+dir+"\\"):
#        dane.append([dir+"\\"+file,str(ind)])
#np.savetxt("train.csv", 
#           dane,
#           delimiter =",", 
#           fmt ='% s')

#def odczytaj_plik(nazwa):
#    plik = pd.read_csv(nazwa, delimiter=",")
#    dane = [tuple(linia) for linia in plik.values]   
#    klasa = [linia[6] for linia in dane]
#    sciezka = [linia[7] for linia in dane]
#    for s in sciezka:
#        s.replace("/", "\\")
#    return klasa, sciezka

#kl, ph = odczytaj_plik("test.csv")

#dane=[]
#klasa=[]
#index0=0
#index1=0
#index2=0
#for i in range(1050):
#    image = cv2.imread(os.path.join(path,ph[i]))
#    newimage = cv2.resize(image,(32, 32))  
#    if kl[i] in {9, 1, 2, 3, 4, 5, 7, 9, 10, 15, 16, 17}:#f
#        klasa.append(1)
#        cv2.imwrite(os.path.join(path+"test\\",f"1_{index1}.png"), newimage)
#        index1+=1
#    elif kl[i] in {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}:#w
#        klasa.append(0)
#        cv2.imwrite(os.path.join(path+"test\\",f"0_{index0}.png"), newimage)
#        index0+=1
#    elif kl[i] in {33, 34, 35, 36, 37, 38, 39, 40}:#n
#        klasa.append(2)
#        cv2.imwrite(os.path.join(path+"test\\",f"2_{index2}.png"), newimage)
#        index2+=1      

#dane = []
#for file in os.listdir(path+"test"+"\\"):
#    klasa = file[0]
#    dane.append(["test"+"\\"+file, klasa])

#np.savetxt("test1.csv", 
#           dane,
#           delimiter =",", 
#           fmt ='% s')
#plik = pd.read_csv("temp.csv", delimiter=",")
#dane = [tuple(linia) for linia in plik.values]
#zrodlo = [linia[0] for linia in dane]
#klasa = [linia[1] for linia in dane]
#piksele=[]
#for sciezka in zrodlo:
#    image = Image.open(os.path.normpath(os.path.join(path,sciezka)))
#    image = image.resize((32, 32))
#    image.save(os.path.normpath(os.path.join(path,sciezka)))
#    #cv2.imwrite(os.path.join(os.path.join(path,sciezka)), image)
