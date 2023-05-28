from load_data import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import *
from keras.optimizers import Adam
from keras import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

###
load_data.N_OF_CLASSES = 3#36
##
print(f"liczba klas = {get_classes_num()}")
print("wczytywanie")
(X_train, y_train), (X_test, y_test) = load_to_network()
print("start")



#model = load_model('Modele/4nsiec36')


model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(load_data.N_OF_CLASSES))


model.compile(optimizer=Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=50,#80
                   validation_data=(X_test, y_test))
model.save('Modele/5nsiec3')


_, avg = model.evaluate(X_test, y_test)
print('Accuracy: {0:.4f}'.format(avg))

y_pred = np.argmax(model.predict(X_test), axis=-1)
confmat = confusion_matrix(y_test, y_pred)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        print(confmat[i][j], end=" ")
    print()
row_sums=confmat.sum(axis=1,keepdims=True)
norm_mat=confmat/row_sums
norm_mat*=100
plt.matshow(norm_mat, cmap=plt.cm.Blues)
plt.title("Sieć konwolucyjna dla {0} klas\nDokładność = {1:.2%}\n".format(get_classes_num(),avg))
plt.ylabel("Rzeczywiste klasy")
plt.xlabel("Przewidywane klasy")
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        plt.text(j, i, f"{int(confmat[i, j])}", ha="center", va="center", color="w")
plt.colorbar(format='%.0f%%')
plt.clim(0, 100)
plt.show()


np.fill_diagonal(norm_mat,0)
plt.matshow(norm_mat, cmap=plt.cm.Greys)
plt.title("Błędy popełnione przy klasyfikacji")
plt.ylabel("Rzeczywiste klasy")
plt.xlabel("Przewidywane klasy")
for i in range(norm_mat.shape[0]):
    for j in range(norm_mat.shape[1]):
        if j == i:
            plt.text(j, i, f"0", ha="center", va="center", color="w")
        else:
            plt.text(j, i, f"{int(confmat[i, j])}", ha="center", va="center", color="w")
plt.colorbar(format='%.1f%%')
plt.show()
pred = confmat.diagonal()/confmat.sum(axis=1)
for i in [i for i in range(get_classes_num())]:
    print("Klasa [{0}] = {1:.4f}".format(i,pred[i]))


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


plt.plot(history.epoch, history.history['accuracy'], 'g', label="Zbiór treningowy")
plt.plot(history.epoch, history.history['val_accuracy'], 'b', label="Zbiór testowy")
plt.title("Dokładność w kolejnych epokach")
plt.xlabel("Epoki")
plt.ylabel("Dokładność")
plt.legend()
plt.figure()
plt.plot(history.epoch, history.history['loss'], 'g', label="Zbiór treningowy")
plt.plot(history.epoch, history.history['val_loss'], 'b', label="Zbiór testowy")
plt.title("Wartość funkcji straty w kolejnych epokach")
plt.xlabel("Epoki")
plt.ylabel("Funkcja straty")
plt.legend()
plt.show()

#tf.keras.utils.plot_model(model, to_file="4cnnNWE.png", show_shapes=True)







def zapisz(nazwa, obj):
    with open(f'{nazwa}', 'wb') as f:
        pickle.dump(obj, f)
    return obj

def wczytaj(nazwa,obj):
    with open(f'{nazwa}', 'rb') as f:
        obj = pickle.load(f)
    return obj

i = zapisz("iters3", history.epoch)
ls = zapisz("trloss3", history.history['loss'])
vls = zapisz("valloss3", history.history['val_loss'])
aTr = zapisz("accTrain3", history.history['accuracy'])
aVl = zapisz("accTest3", history.history['val_accuracy'])