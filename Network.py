from load_data import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras import *
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
###
load_data.N_OF_CLASSES = 3#36
##
print(f"liczba klas = {get_classes_num()}")
print("wczytywanie")
(X_train, y_train), (X_test, y_test) = load_to_network()
print("start")



model = tf.keras.models.load_model('Modele/siec3')
#model = Sequential([
#  layers.Rescaling(1./255, input_shape=(32, 32, 3)),
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Flatten(),
#  layers.Dense(128, activation='relu'),
#  layers.Dense(load_data.N_OF_CLASSES)
#])


#model.compile(optimizer=Adam(),
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])


#history = model.fit(X_train, y_train, epochs=20, #80
#                   validation_data=(X_test, y_test))
#model.save('Modele/siec3')


_, avg = model.evaluate(X_test, y_test)
print('Accuracy: {0:.4f}'.format(avg))



#plt.plot(history.epoch, history.history["loss"], 'g')
#plt.title("Wartość funkcji straty w kolejnych epokach")
#plt.xlabel("Epoki")
#plt.ylabel("Funkcja straty")
#plt.show()

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