import numpy as np
from matplotlib import pyplot as plt
from kNN import NearestNeighborClasiffier
from Linear import LinearClasiffier
from load_data import load_data, get_classes_num
import pickle, json

###
load_data.N_OF_CLASSES = 36#36
#

def zapisz(nazwa, obj):
    with open(f'{nazwa}', 'wb') as f:
        pickle.dump(obj, f)
    return obj

def wczytaj(nazwa,obj):
    with open(f'{nazwa}', 'rb') as f:
        obj = pickle.load(f)
    return obj

print(f"liczba klas = {get_classes_num()}")
print("wczytywanie")
train_data, test_data = load_data()
print("start")

###model = NearestNeighborClasiffier(klasy=get_classes_num(),norma = 2, k = 1)
model = LinearClasiffier(klasy=get_classes_num(), iters=10000,step=0.001)

#model = LinearClasiffier(klasy=get_classes_num(), iters=1000,step=0.0001)

#model.wczytaj_model("./Modele/model_3_10000_NEW")

model.train(*train_data,*test_data)
avg,confmat = model.evaluate(*test_data)

model.zapisz_model("modelLS36")

i, ls, vls = model.get_loss()
i, aTr, aVl = model.get_acc()

#i, ls, vls = [],[],[]
#i, aTr, aVl = [],[],[]

i = zapisz("iters", i)
ls = zapisz("trloss2", ls)
vls = zapisz("valloss2", vls)
aTr = zapisz("accTrain2", aTr)
aVl = zapisz("accTest2", aVl)

plt.plot(i, ls,'b',label="Zbiór treningowy")
plt.title("Wartość funkcji straty w kolejnych iteracjach")
plt.ylabel("Strata")
plt.xlabel("Iteracje")
plt.show()

plt.plot(i, vls,'r',label="Zbiór testowy")
plt.title("Wartość funkcji straty w kolejnych iteracjach")
plt.ylabel("Strata")
plt.xlabel("Iteracje")
#plt.legend(loc="upper right")
plt.show()

plt.figure()

plt.plot(i, aTr,label="Zbiór treningowy")
plt.plot(i, aVl, label="Zbiór testowy")
plt.title("Dokładność kolejnych iteracjach")
plt.ylabel("Dokładność")
plt.xlabel("Iteracje")
plt.legend(loc="lower right")
plt.show()



row_sums=confmat.sum(axis=1,keepdims=True)
norm_mat=confmat/row_sums
norm_mat*=100

plt.matshow(norm_mat, cmap=plt.cm.Blues)
plt.title("Klasyfikator liniowy dla {0} klas\nDokładność = {1:.2%}\n".format(get_classes_num(),avg))
#plt.title("Klasyfikator kNN dla {0} klas\nDokładność = {1}%\n".format(get_classes_num(),avg*100))
plt.ylabel("Rzeczywiste klasy")
plt.xlabel("Przewidywane klasy")
#for i in range(confmat.shape[0]):
#    for j in range(confmat.shape[1]):
#        plt.text(j, i, f"{int(confmat[i, j])}", ha="center", va="center", color="w")
plt.colorbar(format='%.0f%%')
plt.clim(0, 100)
plt.show()


np.fill_diagonal(norm_mat,0)
plt.matshow(norm_mat, cmap=plt.cm.Greys)
plt.title("Błędy popełnione przy klasyfikacji")
plt.ylabel("Rzeczywiste klasy")
plt.xlabel("Przewidywane klasy")
#for i in range(norm_mat.shape[0]):
#    for j in range(norm_mat.shape[1]):
#        if j == i:
#            plt.text(j, i, f"0", ha="center", va="center", color="w")
#        else:
#            plt.text(j, i, f"{int(confmat[i, j])}", ha="center", va="center", color="w")
plt.colorbar(format='%.2f%%')
plt.show()

def prf1a(confmat):
    true_positives = np.diag(confmat)
    false_positives = np.sum(confmat, axis=1) - true_positives
    false_negatives = np.sum(confmat, axis=0) - true_positives
    
    precision = round(np.mean(true_positives / (true_positives + false_positives)),4)
    recall = round(np.mean(true_positives / (true_positives + false_negatives)),4)
    f1_score = round(np.mean(2 * (precision * recall) / (precision + recall)),4)
    accuracy = round(np.sum(true_positives) / np.sum(confmat),4)
    
    return precision, recall, f1_score, accuracy

precision, recall, f1_score, accuracy = prf1a(confmat)
print("Dokładność:", accuracy)
print("Precyzja:", precision)
print("Zwrot:", recall)
print("Miara F1:", f1_score)

###############
#print("liczenie wg k")

#output = []
#for i in [3, 36]:
#    load_data.N_OF_CLASSES = i
#    print(f"liczba klas = {get_classes_num()}")
#    print("wczytywanie")
#    train_data, test_data = load_data()
#    print("start")
#    for k in [3, 5]:
#        model = NearestNeighborClasiffier(klasy=i, k=k, norma=2)
#        model.train(*train_data)
#        avg,_ = model.evaluate(*test_data)
#        print(f"k={k} n=2 avg={avg}")
#        output.append(f"{k},2,{avg}")
#print("k,n,avg")
#for o in output:
#    print(o)
