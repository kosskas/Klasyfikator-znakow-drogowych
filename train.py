import numpy as np
from matplotlib import pyplot as plt
from kNN import NearestNeighborClasiffier
from Linear import LinearClasiffier
from load_data import load_data, get_classes_num

###
#load_data.N_OF_CLASSES = 36#36
###
#print(f"liczba klas = {get_classes_num()}")
#print("wczytywanie")
#train_data, test_data = load_data()
#print("start")


#model = NearestNeighborClasiffier(klasy=get_classes_num(),norma = 3, k = 1)

#model = LinearClasiffier(klasy=get_classes_num(), metoda="localsearch",iters=1000,step=0.0001)
#model.wczytaj_model("./Modele/model_3_100000")

#model.train(*train_data)
#avg,confmat = model.evaluate(*test_data)
#model.zapisz_model("nowy_model")






##############
print("liczenie wg k")
output = []
for i in [3, 36]:
    load_data.N_OF_CLASSES = i
    print(f"liczba klas = {get_classes_num()}")
    print("wczytywanie")
    train_data, test_data = load_data()
    print("start")
    for k in [7, 10, 20]:
        print(f"\n --- i = {i}  k = {k} ---")
        model = NearestNeighborClasiffier(klasy=i, k=k, norma=1)
        model.train(*train_data)
        avg,_ = model.evaluate(*test_data)
        print(f"k={k} n=1 avg={avg}")
        output.append(f"{k},1,{avg}")
print("k,n,avg")
for o in output:
    print(o)


# print("liczenie wg norm")
# output = []
# for i in [3, 36]:
#     load_data.N_OF_CLASSES = i
#     print(f"\nliczba klas = {get_classes_num()}")
#     print("wczytywanie")
#     train_data, test_data = load_data()
#     print("start")
#     for norma in [4, 5, 6]:
#         print(f"\n --- i = {i}  n = {norma} ---")
#         model = NearestNeighborClasiffier(klasy=i, k=1, norma=norma)
#         model.train(*train_data)
#         avg,_ = model.evaluate(*test_data)
#         print(f"k=1 n={norma} avg={avg}")
#         output.append(f"1,{norma},{avg}")
# print("k,n,avg")
# for o in output:
#     print(o)
