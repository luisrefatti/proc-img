from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageTk
import numpy as np

# ===================== image processing functions =====================


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


def toggle_inputs(event=None):
    operation = operations.get()
    constant_ops = ["Add Constant", "Subtract Constant",
                    "Multiply by Constant", "Divide by Constant", "Thresholding"]
    alpha_ops = ["Linear Blend"]

    constant_entry.configure(
        state="normal" if operation in constant_ops else "disabled")
    alpha_entry.configure(
        state="normal" if operation in alpha_ops else "disabled")

    if operation not in constant_ops:
        constant_entry.delete(0, 'end')
    if operation not in alpha_ops:
        alpha_entry.delete(0, 'end')

# math operations


def add_images():
    validate_two_images()
    return np.clip(image1_array.astype(int) + image2_array.astype(int), 0, 255).astype(np.uint8)


def subtract_images():
    validate_two_images()
    return np.clip(image1_array.astype(int) - image2_array.astype(int), 0, 255).astype(np.uint8)


def image_difference():
    validate_two_images()
    return np.abs(image1_array.astype(int) - image2_array.astype(int)).astype(np.uint8)


def apply_constant_operation(operation, value):
    validate_one_image()
    if operation == 'add':
        return np.clip(image1_array.astype(int) + value, 0, 255).astype(np.uint8)
    elif operation == 'subtract':
        return np.clip(image1_array.astype(int) - value, 0, 255).astype(np.uint8)
    elif operation == 'multiply':
        return np.clip(image1_array.astype(float) * value, 0, 255).astype(np.uint8)
    elif operation == 'divide':
        return np.clip(image1_array.astype(float) / value, 0, 255).astype(np.uint8)

# geometric operations


def flip_horizontal():
    validate_one_image()
    return np.fliplr(image1_array).copy()


def flip_vertical():
    validate_one_image()
    return np.flipud(image1_array).copy()

# color conversion


def rgb_to_grayscale():
    validate_one_image()
    return np.dot(image1_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

# blend operations


def linear_blend(alpha):
    validate_two_images()
    return (alpha * image1_array + (1 - alpha) * image2_array).astype(np.uint8)


def average_images():
    return linear_blend(0.5)

# logic operations


def logical_operation(operation):
    validate_one_image()
    img1_bin = binarize_image(image1_array)

    if operation != "NOT":
        validate_two_images()
        img2_bin = binarize_image(image2_array)

    if operation == "AND":
        return np.bitwise_and(img1_bin, img2_bin)
    elif operation == "OR":
        return np.bitwise_or(img1_bin, img2_bin)
    elif operation == "XOR":
        return np.bitwise_xor(img1_bin, img2_bin)
    elif operation == "NOT":
        return np.bitwise_not(img1_bin)


def binarize_image(img, threshold=127):
    gray = rgb_to_grayscale() if len(img.shape) == 3 else img
    return np.where(gray > threshold, 255, 0).astype(np.uint8)

# histogram equalization


def histogram_equalization():
    validate_one_image()
    if len(image1_array.shape) == 3:
        yuv = rgb_to_yuv(image1_array)
        yuv[:, :, 0] = equalize_channel(yuv[:, :, 0])
        return yuv_to_rgb(yuv)
    else:
        return equalize_channel(image1_array)


def equalize_channel(channel):
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape).astype(np.uint8)


def rgb_to_yuv(rgb):
    yuv = np.empty_like(rgb)
    yuv[..., 0] = 0.299 * rgb[..., 0] + 0.587 * \
        rgb[..., 1] + 0.114 * rgb[..., 2]  # Y
    yuv[..., 1] = -0.14713 * rgb[..., 0] - 0.28886 * \
        rgb[..., 1] + 0.436 * rgb[..., 2]  # U
    yuv[..., 2] = 0.615 * rgb[..., 0] - 0.51499 * \
        rgb[..., 1] - 0.10001 * rgb[..., 2]  # V
    return yuv


def yuv_to_rgb(yuv):
    rgb = np.empty_like(yuv)
    rgb[..., 0] = yuv[..., 0] + 1.13983 * yuv[..., 2]  # R
    rgb[..., 1] = yuv[..., 0] - 0.39465 * \
        yuv[..., 1] - 0.58060 * yuv[..., 2]  # G
    rgb[..., 2] = yuv[..., 0] + 2.03211 * yuv[..., 1]  # B
    return np.clip(rgb, 0, 255).astype(np.uint8)

# thresold / limiarização


def threshold_image(threshold):
    validate_one_image()
    gray = rgb_to_grayscale() if len(image1_array.shape) == 3 else image1_array
    return np.where(gray > threshold, 255, 0).astype(np.uint8)

# aux functions (expections/errors)


def validate_one_image():
    if image1_array is None:
        raise ValueError("Load an image first")


def validate_two_images():
    if image1_array is None or image2_array is None:
        raise ValueError("Load both images first")
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have same dimensions")


def apply_operation():
    global result_image
    operation = operations.get()
    constant = constant_entry.get()

    try:
        if operation == "RGB to Grayscale":
            result = rgb_to_grayscale()
        elif operation == "Flip Horizontal":
            result = flip_horizontal()
        elif operation == "Flip Vertical":
            result = flip_vertical()
        elif operation == "Image Difference":
            result = image_difference()
        elif operation == "Linear Blend":
            alpha = float(alpha_entry.get()) if alpha_entry.get() else 0.5
            result = linear_blend(alpha)
        elif operation == "Average Images":
            result = average_images()
        elif operation in ["AND (Binary)", "OR (Binary)", "XOR (Binary)", "NOT (Binary)"]:
            result = logical_operation(operation.split()[0])
        elif operation == "Histogram Equalization":
            result = histogram_equalization()
        elif operation == "Thresholding":
            threshold = int(constant) if constant else 127
            result = threshold_image(threshold)
        else:
            if "Add" in operation:
                result = add_images() if "Images" in operation else apply_constant_operation(
                    'add', float(constant))
            elif "Subtract" in operation:
                result = subtract_images() if "Images" in operation else apply_constant_operation(
                    'subtract', float(constant))
            elif "Multiply" in operation:
                result = apply_constant_operation('multiply', float(constant))
            elif "Divide" in operation:
                result = apply_constant_operation('divide', float(constant))

        if len(result.shape) == 2:
            result_image = Image.fromarray(result, 'L')
        else:
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


# ===================== gui =====================
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title('Image Processing | by luisrefatti')
root.geometry('1280x720')

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

image1 = None
image2 = None
result_image = None
image1_array = None
image2_array = None

# Cabeçalho
app_name_label = customtkinter.CTkLabel(root,
                                        text="Image Processing App",
                                        font=("Montserrat", 15, "bold"),
                                        text_color="#029cff")
app_name_label.grid(row=0, column=1, padx=10, pady=10)

app_pipe_label = customtkinter.CTkLabel(root,
                                        text="|",
                                        font=("Montserrat", 12),
                                        text_color="gray")
app_pipe_label.grid(row=0, column=2, padx=10, pady=10)

app_author_label = customtkinter.CTkLabel(root,
                                          text="Created by Luis Fernando Refatti Boff",
                                          font=("Montserrat", 12),
                                          text_color="white")
app_author_label.grid(row=0, column=3, padx=10)

# main content
main_frame = customtkinter.CTkFrame(root)
main_frame.grid(row=1, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")

main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# load section
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

# preview
image_frame = customtkinter.CTkFrame(main_frame)
image_frame.pack(pady=20, fill="both", expand=True)

img1_label = customtkinter.CTkLabel(image_frame, text="Image 1 Preview")
img1_label.grid(row=0, column=0, padx=20, sticky="nsew")

img2_label = customtkinter.CTkLabel(image_frame, text="Image 2 Preview")
img2_label.grid(row=0, column=1, padx=20, sticky="nsew")

# control
controls_frame = customtkinter.CTkFrame(main_frame)
controls_frame.pack(pady=10, fill="x")

blend_frame = customtkinter.CTkFrame(controls_frame)
blend_frame.pack(side="left", padx=5)

lbl_alpha = customtkinter.CTkLabel(blend_frame, text="Alpha:")
lbl_alpha.pack(side="left")

alpha_entry = customtkinter.CTkEntry(blend_frame, width=50, state="disabled")
alpha_entry.pack(side="left", padx=5)

operations = customtkinter.CTkOptionMenu(
    controls_frame,
    values=[
        "Add Images",
        "Add Constant",
        "Subtract Images",
        "Subtract Constant",
        "Multiply by Constant",
        "Divide by Constant",
        "RGB to Grayscale",
        "Flip Horizontal",
        "Flip Vertical",
        "Image Difference",
        "Linear Blend",
        "Average Images",
        "AND (Binary)",
        "OR (Binary)",
        "XOR (Binary)",
        "NOT (Binary)",
        "Histogram Equalization",
        "Thresholding"
    ],
    command=toggle_inputs
)
operations.pack(side="left", padx=10)

constant_frame = customtkinter.CTkFrame(controls_frame)
constant_frame.pack(side="left", padx=5)

constant_label = customtkinter.CTkLabel(
    constant_frame, text="Constant/Threshold:")
constant_label.pack(side="left")

constant_entry = customtkinter.CTkEntry(
    constant_frame,
    width=120,
    state="disabled"
)
constant_entry.pack(side="left", padx=5)

btn_apply = customtkinter.CTkButton(controls_frame,
                                    text="Apply Operation",
                                    command=apply_operation)
btn_apply.pack(side="left", padx=10)

# result
result_frame = customtkinter.CTkFrame(main_frame)
result_frame.pack(pady=20, fill="both", expand=True)

result_label = customtkinter.CTkLabel(result_frame, text="Result Preview")
result_label.pack(expand=True)

btn_save = customtkinter.CTkButton(result_frame,
                                   text="Save Result",
                                   command=save_result)
btn_save.pack(pady=10)

toggle_inputs()

root.mainloop()
