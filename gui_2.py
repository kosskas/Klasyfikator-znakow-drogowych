# Import customtkinter module
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
import tkinter

import customtkinter as ctk
import tensorflow as tf
from PIL import Image

import os, sys, csv, cv2, random
import numpy as np
from Linear import LinearClasiffier
from kNN import NearestNeighborClasiffier
from load_data import load_data, get_classes_num, load_images, loadFast

load_data.N_OF_CLASSES = 36  # 36


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
             "Zwężenie jezdni - prawostronne",
             "Roboty na drodze",
             "Sygnały świetlne",
             "Przejście dla pieszych",
             "Dzieci",
             "Rowerzyści",
             "Oszronienie jezdni",
             "Dzikie zwierzęta",
             "Nakaz jazdy w prawo za znakiem",
             "Nakaz jazdy w lewo za znakiem",
             "Nakaz jazdy prosto",
             "Nakaz jazdy prosto lub w prawo",
             "Nakaz jazdy prosto lub w lewo",
             "Nakaz jazdy z prawej strony znaku",
             "Nakaz jazdy z lewej strony znaku",
             "Ruch okrężny"]


# linear_classifier
print("loading linear_classifier_model...")
linear_classifier_model = LinearClasiffier(klasy=get_classes_num(), iters=10000, step=0.00001)
linear_classifier_model.wczytaj_model("Modele/model_lin36_100000_LS")
print("done")

# convolutional_network
print("loading convolutional_network_model...")
convolutional_network_model = tf.keras.models.load_model("Modele/nsiec36_4")
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


class ImageFrame(ctk.CTkFrame):
    def __init__(self,
                 *args,
                 header_name="Image Frame",
                 image_h, image_w,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.headerName = header_name
        self.imageHeight = image_h
        self.imageWidth = image_w

        # header
        if self.headerName != "":
            self.header = ctk.CTkLabel(self, text=self.headerName)
            self.header.grid(row=0, column=0, pady=(8, 0))

        # image
        self.imageLabel = ctk.CTkLabel(self, text="", fg_color="transparent")
        self.imageLabel.grid(row=1, column=0, padx=13, pady=13)
        self.set_image(None)

    def set_image(self, image_path):
        if image_path is None:
            image_path = f".\\Meta\\none.png"

        self.imageLabel = ctk.CTkLabel(self,
                                       image=ctk.CTkImage(
                                           light_image=Image.open(image_path),
                                           dark_image=Image.open(image_path),
                                           size=(self.imageHeight, self.imageWidth)
                                       ),
                                       text="")
        self.imageLabel.grid(row=1, column=0, padx=13, pady=13)


class ImageWithDescriptionFrame(ctk.CTkFrame):
    def __init__(self,
                 *args,
                 header_name="Image With Description Frame",
                 image_h,
                 image_w,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.headerName = header_name

        # image
        self.imageLabel = ImageFrame(self, header_name="", fg_color="transparent", image_h=image_h, image_w=image_w)
        self.imageLabel.grid(row=0, column=0)

        # textbox
        self.textbox = ctk.CTkTextbox(master=self, height=89, width=288, font=('Helvetica', 13))
        self.textbox.grid(row=0, column=1, padx=(0, 13), pady=13)

    def set(self, image_path, description):

        self.imageLabel.set_image(image_path)

        if description is None:
            self.textbox.delete("0.0", "end")
        else:
            self.textbox.insert("0.0", description)


class AnswerFrame(ctk.CTkFrame):
    def __init__(self,
                 *args,
                 header_name="Answer Frame",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.headerName = header_name

        # header
        self.header = ctk.CTkLabel(self, width=288, text=self.headerName, font=('Helvetica', 21))
        self.header.grid(row=0, column=0, padx=13, pady=13)

        # image & desc
        self.imageWithDesc = ImageWithDescriptionFrame(self,
                                                       header_name="Answer 1",
                                                       image_h=100, image_w=100,
                                                       fg_color="transparent"
                                                       )
        self.imageWithDesc.grid(row=0, column=1)


class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, window_width, window_height, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

        self.title("Klasyfikator znaków drogowych")
        self.geometry(f"{window_width}x{window_height}")

        # context
        self.imagePath = None
        self.mode = tkinter.StringVar(value="36")

        # choose mode
        self.frame = ctk.CTkFrame(master=self)
        self.frame.grid(row=0, column=0, padx=13, pady=13)
        self.button3 = ctk.CTkRadioButton(self.frame,
                                          variable=self.mode,
                                          command=self.answer,
                                          value="3",
                                          text="3 klasy")
        self.button3.grid(row=0, column=0, padx=8, pady=8)
        self.button36 = ctk.CTkRadioButton(self.frame,
                                           variable=self.mode,
                                           command=self.answer,
                                           value="36",
                                           text="36 klasy")
        self.button36.grid(row=0, column=1, padx=8, pady=8)

        # choose image Section
        self.chooseSection = ctk.CTkFrame(master=self, fg_color="transparent")
        self.chooseSection.grid(row=1, column=0, padx=13, pady=13)

        self.chooseImageButton = ctk.CTkButton(self.chooseSection,
                                               command=self.choose_image,
                                               text="Wybierz znak \n lub przeciągnij",
                                               width=144, height=55
                                               )
        self.chooseImageButton.grid(row=0, column=0, padx=21, pady=21)
        self.chooseImageButton.drop_target_register(DND_FILES)
        self.chooseImageButton.dnd_bind("<<Drop>>", self.drop_image)

        self.chosenImageLabel = ImageFrame(self.chooseSection,
                                           header_name="Testowany znak",
                                           image_h=100, image_w=100
                                           )
        self.chosenImageLabel.grid(row=0, column=1, padx=21, pady=21)

        # answer Section
        self.answerSection = ctk.CTkFrame(master=self, fg_color="transparent", width=window_width)
        self.answerSection.grid(row=2, column=0, padx=21, pady=13)

        self.kNNAnswerFrame = AnswerFrame(self.answerSection, header_name="kNN")
        self.kNNAnswerFrame.grid(row=0, column=0, pady=(0, 13))

        self.linearClassifierAnswerFrame = AnswerFrame(self.answerSection, header_name="Klasyfikator liniowy")
        self.linearClassifierAnswerFrame.grid(row=1, column=0, pady=(0, 13))

        self.convolutionalNetworkAnswerFrame = AnswerFrame(self.answerSection, header_name="Sieć konwolucyjna")
        self.convolutionalNetworkAnswerFrame.grid(row=2, column=0, pady=(0, 13))

    def drop_image(self, event):
        self.imagePath = event.data.split("{", 1)[1].split("}", 1)[0]
        self.answer()

    def choose_image(self):
        self.imagePath = filedialog.askopenfilename(
            initialdir="./zdjTestowe",
            title="Choose a photo",
            filetypes=(("png", "*.png"), ("jpg", "*.jpg"))
        )
        self.answer()

    def answer(self):
        self.chosenImageLabel.set_image(None)
        self.kNNAnswerFrame.imageWithDesc.set(None, None)
        self.linearClassifierAnswerFrame.imageWithDesc.set(None, None)
        self.convolutionalNetworkAnswerFrame.imageWithDesc.set(None, None)

        if self.imagePath is not None and self.imagePath != "":
            self.chosenImageLabel.set_image(self.imagePath)
            self.show_answer(self.kNNAnswerFrame, predicate("kNN", self.imagePath))
            self.show_answer(self.linearClassifierAnswerFrame, predicate("linear_classifier", self.imagePath))
            self.show_answer(self.convolutionalNetworkAnswerFrame, predicate("convolutional_network", self.imagePath))

    def show_answer(self, answer_frame: AnswerFrame, answer):
        if self.mode.get() == "3":
            answer_frame.imageWithDesc.set(
                None,
                f"Znak {sign_class(answer)}"
            )
        elif self.mode.get() == "36":
            answer_frame.imageWithDesc.set(
                f".\\Meta\\{answer}.png",
                f"Znak {sign_class(answer)} \n\n {sign_desc[answer]}"
            )


if __name__ == "__main__":
    # Sets the appearance mode of the application
    # "System" sets the appearance same as that of the system
    ctk.set_appearance_mode("System")

    # Sets the color of the widgets
    # Supported themes: green, dark-blue, blue
    ctk.set_default_color_theme("dark-blue")

    app = App(window_width=780, window_height=730)

    # Runs the app
    app.mainloop()
