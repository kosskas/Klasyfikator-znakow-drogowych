import numpy as np
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
model = NearestNeighborClasiffier(klasy=get_classes_num(),norma = 2, k=1)
#model = LinearClasiffier(klasy=get_classes_num(), metoda="localsearch",iters=1000,step=0.0001)
##model.wczytaj_model("rr")

model.train(*train_data)
model.evaluate(*test_data)
#model.zapisz_model("36_1000")

#
#i, ls = model.get_loss()
#plt.plot(i, ls)
#plt.title("Wartość funkcji straty w kolejnych iteracjach")
#plt.xlabel("Iteracje")
#plt.ylabel("Funkcja straty")
#plt.show()

###############

#for j in [-5, -6, -7]:
#    step = 10**j
#    print(f"LR = {10**j}")
#    model = LinearClasiffier(klasy=get_classes_num(),metoda="localsearch",iters=10000,step=step)
#    model.train(*train_data)
#    i, ls = model.get_loss()
#    model.evaluate(*test_data)
#    plt.plot(i, ls, label = f"LR = {10**j}")
#plt.legend(loc="upper right")
#plt.title("Wartość funkcji straty w kolejnych iteracjach")
#plt.xlabel("Iteracje")
#plt.ylabel("Funkcja straty")
#plt.show()







#output = []
#for i in [3, 36]:
#    for k in [1, 3, 5,     7, 10, 20,   30, 50, 100]:
#        model = NearestNeighborClasiffier(klasy=i, k=k)
#        model.train(*train_data)
#        avg = model.evaluate(*test_data)
#        print(f"k={k} n={n} avg={avg}")
#        output.append(f"{k},{n},{avg}")
#print("k,n,avg")
#for o in output:
#    print(o)