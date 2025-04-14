from tkinter import *
import customtkinter

root = customtkinter.CTk()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root.title('Image Processing | by luisrefatti')
root.geometry('1920x1080')

my_button = customtkinter.CTkButton(root, text="img-proc-test")
my_button.pack(pady=0)

root.mainloop()
