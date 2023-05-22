from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
from load_data import load_data, get_classes_num
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,accuracy_score

###
load_data.N_OF_CLASSES = 36#36
#
print(f"liczba klas = {get_classes_num()}")
print("wczytywanie")
(X_train, y_train), (X_test, y_test) = load_data()
print("start")
# Tworzenie i trenowanie klasyfikatora SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Przewidywanie klas dla danych testowych
y_pred = classifier.predict(X_test)

# Obliczenie dokładności klasyfikatora
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność klasyfikatora SVM: {accuracy * 100}%")