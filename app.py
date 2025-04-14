from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageTk
import numpy as np


# ===================== FUNCTIONS =====================
def load_image(img_num):
    global image1, image2, image1_array, image2_array
    file_path = filedialog.askopenfilename(filetypes=[
        ("Image Files", "*.bmp *.jpg *.png *.jpeg")
    ])

    if not file_path:
        return

    try:
        img = Image.open(file_path).convert("RGB")
        img_array = np.array(img)

        if img_num == 1:
            image1 = img
            image1_array = img_array
            display_image(img, img1_label)
        else:
            image2 = img
            image2_array = img_array
            display_image(img, img2_label)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")


def display_image(img, label):
    img.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(img)
    label.configure(image=photo)
    label.image = photo


def add_images():
    if image1_array is None or image2_array is None:
        raise ValueError("Load both images first")
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have same dimensions")
    return np.clip(image1_array + image2_array, 0, 255).astype(np.uint8)


def add_constant(value):
    if image1_array is None:
        raise ValueError("Load an image first")
    return np.clip(image1_array + value, 0, 255).astype(np.uint8)


def subtract_images():
    if image1_array is None or image2_array is None:
        raise ValueError("Load both images first")
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have same dimensions")
    return np.clip(image1_array - image2_array, 0, 255).astype(np.uint8)


def subtract_constant(value):
    if image1_array is None:
        raise ValueError("Load an image first")
    return np.clip(image1_array - value, 0, 255).astype(np.uint8)


def multiply_constant(value):
    if image1_array is None:
        raise ValueError("Load an image first")
    return np.clip(image1_array * value, 0, 255).astype(np.uint8)


def divide_constant(value):
    if image1_array is None:
        raise ValueError("Load an image first")
    if value == 0:
        raise ValueError("Cannot divide by zero")
    return np.clip(image1_array / value, 0, 255).astype(np.uint8)


def apply_operation():
    global result_image
    operation = operations.get()
    constant = constant_entry.get()

    try:
        if any(op in operation for op in ["Add", "Subtract", "Multiply", "Divide"]):
            if not constant and "Constant" in operation:
                raise ValueError("Please enter a constant value")
            if constant:
                constant = float(constant)

        if operation == "Add Images":
            result = add_images()
        elif operation == "Add Constant":
            result = add_constant(constant)
        elif operation == "Subtract Images":
            result = subtract_images()
        elif operation == "Subtract Constant":
            result = subtract_constant(constant)
        elif operation == "Multiply by Constant":
            result = multiply_constant(constant)
        elif operation == "Divide by Constant":
            result = divide_constant(constant)
        else:
            raise ValueError("Invalid operation selected")

        result_image = Image.fromarray(result)
        display_image(result_image, result_label)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def save_result():
    if result_image is None:
        messagebox.showwarning("Warning", "No result to save")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
    )

    if file_path:
        result_image.save(file_path)
        messagebox.showinfo("Success", "Image saved successfully!")


# ===================== GUI SETUP =====================
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title('Image Processing | by luisrefatti')

# Configurar expansão da janela
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Global variables
image1 = None
image2 = None
result_image = None
image1_array = None
image2_array = None

# Header Section
app_name_label = customtkinter.CTkLabel(root,
                                        text="Image Processing App",
                                        font=("Montserrat", 15, "bold"),
                                        text_color="#029cff")
app_name_label.grid(row=0, column=1, padx=10, pady=10)

app_pipe_label = customtkinter.CTkLabel(root,
                                        text="|",
                                        font=("Montserrat", 12),
                                        text_color="gray")
app_pipe_label.grid(row=0, column=2, padx=0)

app_author_label = customtkinter.CTkLabel(root,
                                          text="Created by Luis Fernando Refatti Boff",
                                          font=("Montserrat", 12),
                                          text_color="white")
app_author_label.grid(row=0, column=3, padx=10)

# Main Content Frame
main_frame = customtkinter.CTkFrame(root)
main_frame.grid(row=1, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")

# Configurar expansão do frame principal
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# Image Load Section
load_frame = customtkinter.CTkFrame(main_frame)
load_frame.pack(pady=10, fill="x")

btn_load1 = customtkinter.CTkButton(load_frame,
                                    text="Load Image 1",
                                    command=lambda: load_image(1))
btn_load1.grid(row=0, column=0, padx=10)

btn_load2 = customtkinter.CTkButton(load_frame,
                                    text="Load Image 2",
                                    command=lambda: load_image(2))
btn_load2.grid(row=0, column=1, padx=10)

# Image Preview Section
image_frame = customtkinter.CTkFrame(main_frame)
image_frame.pack(pady=20, fill="both", expand=True)

img1_label = customtkinter.CTkLabel(image_frame, text="Image 1 Preview")
img1_label.grid(row=0, column=0, padx=20, sticky="nsew")

img2_label = customtkinter.CTkLabel(image_frame, text="Image 2 Preview")
img2_label.grid(row=0, column=1, padx=20, sticky="nsew")

# Controls Section
controls_frame = customtkinter.CTkFrame(main_frame)
controls_frame.pack(pady=10, fill="x")

operations = customtkinter.CTkOptionMenu(controls_frame,
                                         values=[
                                             "Add Images",
                                             "Add Constant",
                                             "Subtract Images",
                                             "Subtract Constant",
                                             "Multiply by Constant",
                                             "Divide by Constant"
                                         ])
operations.pack(side="left", padx=10)

constant_entry = customtkinter.CTkEntry(controls_frame,
                                        placeholder_text="Constant Value",
                                        width=120)
constant_entry.pack(side="left", padx=10)

btn_apply = customtkinter.CTkButton(controls_frame,
                                    text="Apply Operation",
                                    command=apply_operation)
btn_apply.pack(side="left", padx=10)

# Result Section
result_frame = customtkinter.CTkFrame(main_frame)
result_frame.pack(pady=20, fill="both", expand=True)

result_label = customtkinter.CTkLabel(result_frame, text="Result Preview")
result_label.pack(expand=True)

btn_save = customtkinter.CTkButton(result_frame,
                                   text="Save Result",
                                   command=save_result)
btn_save.pack(pady=10)

root.mainloop()
