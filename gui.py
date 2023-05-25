# Import customtkinter module
import tkinter
from tkinter import filedialog

import customtkinter as ctk
import tensorflow as tf
from PIL import Image

import os, sys, csv, cv2, random
import numpy as np
from Linear import LinearClasiffier
from kNN import NearestNeighborClasiffier
from load_data import load_data, get_classes_num, load_images, loadFast

load_data.N_OF_CLASSES = 36 # 36


def sign_class(idx):
    if 0 <= idx <= 12:
        return "Zakazu"
    if 13 <= idx <= 27:
        return "Ostrzegawczy"
    if 28 <= idx <= 35:
        return "Nakazu"
    return "Unknown class"


sign_desc = ["Ograniczenie prędkości do 20 km/h",
             "Ograniczenie prędkości do 30 km/h",
             "Ograniczenie prędkości do 50 km/h",
             "Ograniczenie prędkości do 60 km/h",
             "Ograniczenie prędkości do 70 km/h",
             "Ograniczenie prędkości do 80 km/h",
             "Ograniczenie prędkości do 100 km/h",
             "Ograniczenie prędkości do 120 km/h",
             "Zakaz wyprzedzania",
             "Zakaz wyprzedzania przez samochody ciężarowe",
             "Zakaz ruchu w obu kierunkach",
             "Zakaz wjazdu samochodów ciężarowych",
             "Zakaz wjazdu",
             "Skrzyżowanie z drogą podporządkowaną",
             "Inne niebezpieczeństwo",
             "Niebezpieczny zakręt w lewo",
             "Niebezpieczny zakręt w prawo",
             "Niebezpieczne zakręty - pierwszy w lewo",
             "Nierówna droga",
             "Śliska jezdnia",
             "Rwężenie jezdni - prawostronne",
             "Roboty na drodze",
             "Sygnały świetlne",
             "Przejście dla pieszych",
             "Dzieci",
             "Rowerzyści!!!",
             "Oszronienie jezdni",
             "Zwierzęta dzikie",
             "Nakaz jazdy w prawo za znakiem",
             "Nakaz jazdy w lewo za znakiem",
             "Nakaz jazdy prosto",
             "Nakaz jazdy prosto lub w prawo",
             "Nakaz jazdy prosto lub w lewo",
             "Nakaz jazdy z prawej strony znaku",
             "Nakaz jazdy z lewej strony znaku",
             "Ruch okrężny"]

# Sets the appearance mode of the application
# "System" sets the appearance same as that of the system
ctk.set_appearance_mode("System")

# Sets the color of the widgets
# Supported themes: green, dark-blue, blue
ctk.set_default_color_theme("green")

# defines
window_width = 600
window_height = 600

# linear_classifier
print("loading linear_classifier_model...")
linear_classifier_model = LinearClasiffier(klasy=get_classes_num(), iters=10000, step=0.00001)
linear_classifier_model.wczytaj_model("Modele/model_36_100000")
print("done")

# convolutional_network
print("loading convolutional_network_model...")
convolutional_network_model = tf.keras.models.load_model("Modele/nsiec36")
print("done")

# kNN
print("loading kNN_model...")
x_train, y_train = loadFast()
kNN_model = NearestNeighborClasiffier(klasy=get_classes_num(), norma=2, k=1)
kNN_model.train(x_train, y_train)
print("done")


def image_to_binary_array(model, image_path):
    pixels = []
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))

    if model == "linear_classifier":
        pixels.append(np.array(image).flatten())
    elif model == "convolutional_network":
        pixels.append(np.array(image))
    elif model == "kNN":
        pixels.append(np.array(image).flatten())

    return np.array(pixels)


def predicate(model, image_path):
    test = image_to_binary_array(model, image_path)
    if model == "linear_classifier":
        return int(linear_classifier_model.predict(test))
    elif model == "convolutional_network":
        return int(np.argmax(convolutional_network_model.predict(test), axis=-1))
    elif model == "kNN":
        return int(kNN_model.predict(test))


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Klasyfikator znaków drogowych")
        self.geometry(f"{window_width}x{window_height}")

        # create 2x2 grid system
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.image_path = None
        self.chosen_image_label = None

        # answer image
        self.answer_image_label = ctk.CTkLabel(self, text="", fg_color="transparent")

        # answer textbox
        self.answer_textbox = ctk.CTkTextbox(master=self, height=100, width=250, corner_radius=3)
        self.answer_textbox.grid(row=1, column=1)

        # frame
        self.frame = ctk.CTkFrame(master=self)
        self.frame.grid(row=0, column=0)

        self.chosen_model = tkinter.StringVar(value="none")

        self.linear_button = ctk.CTkRadioButton(self.frame, variable=self.chosen_model, command=self.show_answer,
                                                  value="linear_classifier", text="klasyfikator liniowy")
        self.linear_button.grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self.convolutional_button = ctk.CTkRadioButton(self.frame, variable=self.chosen_model, command=self.show_answer,
                                                    value="convolutional_network", text="sieć konwolucyjna")
        self.convolutional_button.grid(row=1, column=0, sticky="w", padx=10, pady=10)

        self.kNN_button = ctk.CTkRadioButton(self.frame, variable=self.chosen_model, command=self.show_answer,
                                                  value="kNN", text="kNN")
        self.kNN_button.grid(row=2, column=0, sticky="w", padx=10, pady=10)

        self.chooseImageButton = ctk.CTkButton(self.frame, command=self.choose_image,
                                               text="Wybierz znak", width=120, height=40)
        self.chooseImageButton.grid(row=3, column=0, padx=10, pady=20)

    def choose_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir="./zdjTestowe",
            title="Choose a photo",
            filetypes=(("png", "*.png"), ("jpg", "*.jpg"))
        )
        self.show_image(self.chosen_image_label, self.image_path, 0, 1, 150, 150, "")
        # reset
        self.chosen_model.set(value="none")
        self.show_image(self.answer_image_label, None, 1, 0, 150, 150, "")
        self.answer_textbox.delete("0.0", "end")

    def show_image(self, image_label_ref, image_path, row, col, h, w, text):
        if image_path is None:
            image_path = f".\\Meta\\none.png"

        image_label_ref = ctk.CTkLabel(self,
                                       image=ctk.CTkImage(
                                            light_image=Image.open(image_path),
                                            dark_image=Image.open(image_path),
                                            size=(h, w)
                                       ),
                                       text=text)
        image_label_ref.grid(row=row, column=col)

    def show_answer(self):
        if self.chosen_model.get() == "none" or self.image_path is None:
            return
        answer = predicate(self.chosen_model.get(), self.image_path)
        self.show_image(self.answer_image_label, f".\\Meta\\{answer}.png", 1, 0, 150, 150, "")
        self.answer_textbox.delete("0.0", "end")
        self.answer_textbox.insert("0.0", f"Znak {sign_class(answer)} \n\n {sign_desc[answer]}")


if __name__ == "__main__":
    app = App()
    # Runs the app
    app.mainloop()
