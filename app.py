from tkinter import *
import customtkinter

root = customtkinter.CTk()

# theme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# app title
root.title('Image Processing | by luisrefatti')
root.geometry('1920x1080')

# log image label


def image():
    my_label.configure(text="Image 01 Ready")


# upload button
my_button = customtkinter.CTkButton(
    root, text="Upload Image 01", command=image)
my_button.pack(pady=0)

# log label
my_label = customtkinter.CTkLabel(root, text="Insert Image 01")
my_label.pack(pady=20)

root.mainloop()
