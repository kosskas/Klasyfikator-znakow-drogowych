from load_data import load_data, get_classes_num
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt
###
load_data.N_OF_CLASSES = 36#36
##
print(f"liczba klas = {get_classes_num()}")
print("wczytywanie")
(X_train, Y_train), (X_test, Y_test) = load_data()
print("start")

y_train = to_categorical(Y_train, load_data.N_OF_CLASSES)
y_test = to_categorical(Y_test, load_data.N_OF_CLASSES)

#model = tf.keras.models.load_model('test.h5')

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(32 * 32 * 3,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(load_data.N_OF_CLASSES, activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 3 klasy
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy']) #36 klas
history = model.fit(X_train, y_train,
        epochs=250, batch_size=32,
        validation_data=(X_test, y_test))
model.save('siec36_t.h5')
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: {0:.4f}'.format(accuracy))

plt.plot(history.epoch, history.history["loss"], 'g')
plt.title("Wartość funkcji straty w kolejnych epokach")
plt.xlabel("Epoki")
plt.ylabel("Funkcja straty")
plt.show()