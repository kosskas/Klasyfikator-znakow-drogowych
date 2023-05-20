# Import customtkinter module
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

import os, sys, csv, cv2, random
import numpy as np
from Linear import LinearClasiffier
from load_data import load_data, get_classes_num
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
window_width = 500
window_height = 500

linear_classifier_model = LinearClasiffier(klasy=get_classes_num(), metoda="localsearch", iters=10000, step=0.00001)
linear_classifier_model.wczytaj_model("./Modele/model_36_100000")


def image_to_binary_array(image_path):
    pixels = []
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    pixels.append(np.array(image).flatten())
    return np.array(pixels)


def predicate(model, image_path):
    test = image_to_binary_array(image_path)
    # prd_idx = int(model.predict(test))
    # print(f"Znak {klasa[nadrzedna(prd_idx)]}\nOpis: {opis[prd_idx]}")
    return int(model.predict(test))


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Klasyfikator znaków drogowych")
        self.geometry(f"{window_width}x{window_height}")

        # create 2x2 grid system
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.chooseImageButton = ctk.CTkButton(self, command=self.choose_image,
                                               text="Wybierz znak", width=120, height=40)
        self.chooseImageButton.grid(row=1, column=1)

        self.image_path = None
        self.image_label = None
        self.answer_textbox = None

    def choose_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir="./zdjTestowe",
            title="Choose a photo",
            filetypes=(("png", "*.png"), ("jpg", "*.jpg"))
        )
        self.show_image(self.image_path, 2, 1, 150, 150, "")
        self.show_answer(self.image_path, 1, 2, 100, 250)

    def show_image(self, image_path, row, col, h, w, text):
        self.image_label = ctk.CTkLabel(self,
                                        image=ctk.CTkImage(
                                            light_image=Image.open(image_path),
                                            dark_image=Image.open(image_path),
                                            size=(h, w)),
                                        text=text
                                        )
        self.image_label.grid(row=row, column=col)

    def show_answer(self, image_path, row, col, h, w):
        idx = predicate(linear_classifier_model, image_path)
        self.answer_textbox = ctk.CTkTextbox(master=self, height=h, width=w, corner_radius=3)
        self.answer_textbox.grid(row=row, column=col)
        self.answer_textbox.insert("0.0", f"Znak {sign_class(idx)}\n\n{sign_desc[idx]}")
        self.show_image(f".\\Meta\\{idx}.png", 2, 2, 150, 150, "")


if __name__ == "__main__":
    app = App()
    # Runs the app
    app.mainloop()


# from tkinter import *
# from PIL import ImageTk,Image
# from tkinter import filedialog
#
#
# root = Tk()
# root.title('SI GUI')
#
#
# def open():
#     global image
#     root.filename = filedialog.askopenfilename(
#         initialdir="./zdjTestowe",
#         title="Choose a photo",
#         filetypes=(("png", "*.png"), ("jpg", "*.jpg"))
#     )
#     image = ImageTk.PhotoImage(Image.open(root.filename))
#     Label(image=image).pack()
#
#
# Button(root, text="Open", command=open).pack()
#
#
# root.mainloop()


