from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageTk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ===================== image processing functions =====================


def create_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    sum_val = 0.0

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum_val += kernel[i, j]

    kernel /= sum_val
    return kernel


def apply_gaussian_filter_manual(image, sigma=1.0, kernel_size=5):
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for c in range(3):
            filtered[:, :, c] = apply_gaussian_filter_manual(
                image[:, :, c], sigma, kernel_size)
        return filtered

    kernel = create_gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='symmetric')
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    filtered = np.sum(windows * kernel, axis=(2, 3))
    return np.clip(filtered, 0, 255).astype(np.uint8)


def load_image(img_num):
    global image1, image2, image1_array, image2_array
    file_path = filedialog.askopenfilename(filetypes=[
        ("Image Files", "*.bmp *.jpg *.png *.jpeg *.tif")
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


# Operation groups structure
OPERATION_GROUPS = {
    "Arithmetic": {
        "operations": [
            "Add Images",
            "Subtract Images",
            "Add Constant",
            "Subtract Constant",
            "Multiply by Constant",
            "Divide by Constant"
        ],
        "needs_constant": ["Add Constant", "Subtract Constant", "Multiply by Constant", "Divide by Constant"]
    },
    "Logic": {
        "operations": ["AND", "OR", "XOR", "NOT"],
        "needs_constant": []
    },
    "Geometric": {
        "operations": ["Flip Horizontal", "Flip Vertical"],
        "needs_constant": []
    },
    "Color": {
        "operations": ["RGB to Grayscale", "Thresholding", "Histogram Equalization"],
        "needs_constant": ["Thresholding"]
    },
    "Blending": {
        "operations": ["Linear Blend", "Average Images"],
        "needs_constant": ["Linear Blend"]
    },
    "Filters": {
        "operations": [
            "MAX Filter",
            "MIN Filter",
            "MEAN Filter",
            "MEDIAN Filter",
            "ORDER Filter",
            "Conservative Smooth",
            "Gaussian Filter"
        ],
        "needs_constant": [
            "MAX Filter", "MIN Filter", "MEAN Filter", "MEDIAN Filter",
            "ORDER Filter", "Conservative Smooth", "Gaussian Filter"
        ]
    }
}


def update_operations_dropdown(selected_group):
    operations = OPERATION_GROUPS[selected_group]["operations"]
    operations_dropdown.configure(values=operations)
    operations_dropdown.set(operations[0] if operations else "")
    toggle_inputs()


def toggle_inputs(event=None):
    selected_group = groups_dropdown.get()
    selected_operation = operations_dropdown.get()

    constant_entry.configure(state="disabled")
    alpha_entry.configure(state="disabled")
    constant_entry.delete(0, 'end')
    alpha_entry.delete(0, 'end')

    if selected_group in OPERATION_GROUPS:
        group_data = OPERATION_GROUPS[selected_group]

        if selected_operation in group_data["needs_constant"]:
            constant_entry.configure(state="normal")
            if selected_operation == "Gaussian Filter":
                constant_entry.insert(0, "1.0")
            elif selected_operation == "ORDER Filter":
                constant_entry.insert(0, "3,4")

        if selected_group == "Blending" and selected_operation == "Linear Blend":
            alpha_entry.configure(state="normal")
            alpha_entry.insert(0, "0.5")

# Math operations


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

# Geometric operations


def flip_horizontal():
    validate_one_image()
    return np.fliplr(image1_array).copy()


def flip_vertical():
    validate_one_image()
    return np.flipud(image1_array).copy()

# Color conversion


def rgb_to_grayscale():
    validate_one_image()
    return np.dot(image1_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

# Blend operations


def linear_blend(alpha):
    validate_two_images()
    return (alpha * image1_array + (1 - alpha) * image2_array).astype(np.uint8)


def average_images():
    return linear_blend(0.5)

# Logic operations


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

# Histogram equalization


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
        rgb[..., 1] + 0.114 * rgb[..., 2]
    yuv[..., 1] = -0.14713 * rgb[..., 0] - \
        0.28886 * rgb[..., 1] + 0.436 * rgb[..., 2]
    yuv[..., 2] = 0.615 * rgb[..., 0] - 0.51499 * \
        rgb[..., 1] - 0.10001 * rgb[..., 2]
    return yuv


def yuv_to_rgb(yuv):
    rgb = np.empty_like(yuv)
    rgb[..., 0] = yuv[..., 0] + 1.13983 * yuv[..., 2]
    rgb[..., 1] = yuv[..., 0] - 0.39465 * yuv[..., 1] - 0.58060 * yuv[..., 2]
    rgb[..., 2] = yuv[..., 0] + 2.03211 * yuv[..., 1]
    return np.clip(rgb, 0, 255).astype(np.uint8)

# Thresholding


def threshold_image(threshold):
    validate_one_image()
    gray = rgb_to_grayscale() if len(image1_array.shape) == 3 else image1_array
    return np.where(gray > threshold, 255, 0).astype(np.uint8)

# Spatial filters


def apply_filter(channel, kernel_size, filter_type, order_rank=None):
    pad = (kernel_size - 1) // 2
    padded = np.pad(channel, pad, mode='symmetric')
    windows = sliding_window_view(padded, (kernel_size, kernel_size))

    if filter_type == 'max':
        filtered = np.max(windows, axis=(2, 3))
    elif filter_type == 'min':
        filtered = np.min(windows, axis=(2, 3))
    elif filter_type == 'mean':
        filtered = np.mean(windows, axis=(2, 3))
    elif filter_type == 'median':
        filtered = np.median(windows, axis=(2, 3))
    elif filter_type == 'order':
        sorted_windows = np.sort(windows.reshape(
            *windows.shape[:2], windows.shape[2]*windows.shape[3]), axis=2)
        filtered = sorted_windows[:, :, order_rank]
    elif filter_type == 'conservative':
        min_vals = np.min(windows, axis=(2, 3))
        max_vals = np.max(windows, axis=(2, 3))
        original = windows[:, :, pad, pad]
        min_diff = np.abs(min_vals - original)
        max_diff = np.abs(max_vals - original)
        filtered = np.where(min_diff < max_diff, min_vals, max_vals)
    else:
        raise ValueError("Invalid filter type")

    return filtered.astype(np.uint8)

# Validation functions


def validate_one_image():
    if image1_array is None:
        raise ValueError("Load an image first")


def validate_two_images():
    if image1_array is None or image2_array is None:
        raise ValueError("Load both images first")
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have same dimensions")

# Main operation handler


def apply_operation():
    global result_image
    selected_group = groups_dropdown.get()
    operation = operations_dropdown.get()
    constant = constant_entry.get()

    try:
        if selected_group == "Select Category" or operation == "Select Operation":
            raise ValueError("Please select both a category and an operation")

        # Group-based processing
        if selected_group == "Arithmetic":
            if "Images" in operation:
                validate_two_images()
                if operation == "Add Images":
                    result = add_images()
                elif operation == "Subtract Images":
                    result = subtract_images()
            else:
                validate_one_image()
                op_type = operation.split()[0].lower()
                value = float(constant) if constant else 0
                result = apply_constant_operation(op_type, value)

        elif selected_group == "Logic":
            result = logical_operation(operation)

        elif selected_group == "Geometric":
            validate_one_image()
            if operation == "Flip Horizontal":
                result = flip_horizontal()
            elif operation == "Flip Vertical":
                result = flip_vertical()

        elif selected_group == "Color":
            validate_one_image()
            if operation == "RGB to Grayscale":
                result = rgb_to_grayscale()
            elif operation == "Thresholding":
                threshold = int(constant) if constant else 127
                result = threshold_image(threshold)
            elif operation == "Histogram Equalization":
                result = histogram_equalization()

        elif selected_group == "Blending":
            validate_two_images()
            if operation == "Linear Blend":
                alpha = float(alpha_entry.get()) if alpha_entry.get() else 0.5
                result = linear_blend(alpha)
            elif operation == "Average Images":
                result = average_images()

        elif selected_group == "Filters":
            validate_one_image()
            kernel_size = 3
            order_rank = None
            sigma = 1.0

            if constant:
                if operation == "Gaussian Filter":
                    sigma = float(constant)
                    kernel_size = min(15, 2 * int(3 * sigma) + 1)
                elif operation == "ORDER Filter":
                    parts = constant.split(',')
                    if len(parts) == 2:
                        kernel_size = int(parts[0])
                        order_rank = int(parts[1])
                    else:
                        raise ValueError("Use 'kernel_size,rank' format")
                else:
                    kernel_size = int(constant)

            filter_type = operation.split()[0].lower()
            if filter_type == 'gaussian':
                result = apply_gaussian_filter_manual(
                    image1_array, sigma, kernel_size)
            else:
                if len(image1_array.shape) == 3:
                    filtered = np.zeros_like(image1_array)
                    for c in range(3):
                        filtered[:, :, c] = apply_filter(
                            image1_array[:, :, c], kernel_size,
                            filter_type, order_rank
                        )
                else:
                    filtered = apply_filter(
                        image1_array, kernel_size,
                        filter_type, order_rank
                    )
                result = filtered.astype(np.uint8)

        # Display result
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


# ===================== GUI setup =====================
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title('Image Processing | by luisrefatti')
root.geometry('1280x720')

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Global variables
image1 = None
image2 = None
result_image = None
image1_array = None
image2_array = None

# Header
header_frame = customtkinter.CTkFrame(root)
header_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

app_name_label = customtkinter.CTkLabel(header_frame, text="Image Processing App",
                                        font=("Montserrat", 15, "bold"),
                                        text_color="#029cff")
app_name_label.pack(side="left", padx=10)

app_name_label = customtkinter.CTkLabel(header_frame, text=" | ",
                                        font=("Montserrat", 15, "bold"),
                                        text_color="#f2f2f2")
app_name_label.pack(side="left", padx=10)

app_author_label = customtkinter.CTkLabel(header_frame, text="Created by Luis Fernando Refatti Boff",
                                          font=("Montserrat", 12),
                                          text_color="white")
app_author_label.pack(side="left", padx=10)

# Main content
main_frame = customtkinter.CTkFrame(root)
main_frame.grid(row=1, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# Load section
load_frame = customtkinter.CTkFrame(main_frame)
load_frame.pack(pady=10, fill="x")

btn_load1 = customtkinter.CTkButton(load_frame, text="Load Image 1",
                                    command=lambda: load_image(1))
btn_load1.grid(row=0, column=0, padx=10)

btn_load2 = customtkinter.CTkButton(load_frame, text="Load Image 2",
                                    command=lambda: load_image(2))
btn_load2.grid(row=0, column=1, padx=10)

# Image previews
image_frame = customtkinter.CTkFrame(main_frame)
image_frame.pack(pady=20, fill="both", expand=True)

img1_label = customtkinter.CTkLabel(image_frame, text="Image 1 Preview")
img1_label.grid(row=0, column=0, padx=20, sticky="nsew")

img2_label = customtkinter.CTkLabel(image_frame, text="Image 2 Preview")
img2_label.grid(row=0, column=1, padx=20, sticky="nsew")

# Controls
controls_frame = customtkinter.CTkFrame(main_frame)
controls_frame.pack(pady=10, fill="x")

groups_dropdown = customtkinter.CTkOptionMenu(
    controls_frame,
    values=list(OPERATION_GROUPS.keys()),
    command=update_operations_dropdown
)
groups_dropdown.set("Select Category")
groups_dropdown.pack(side="left", padx=5)

operations_dropdown = customtkinter.CTkOptionMenu(
    controls_frame,
    values=[],
    command=toggle_inputs
)
operations_dropdown.set("Select Operation")
operations_dropdown.pack(side="left", padx=5)

constant_frame = customtkinter.CTkFrame(controls_frame)
constant_frame.pack(side="left", padx=5)

constant_label = customtkinter.CTkLabel(constant_frame, text="Parameter:")
constant_label.pack(side="left")

constant_entry = customtkinter.CTkEntry(
    constant_frame, width=120, state="disabled")
constant_entry.pack(side="left", padx=5)

blend_frame = customtkinter.CTkFrame(controls_frame)
blend_frame.pack(side="left", padx=5)

lbl_alpha = customtkinter.CTkLabel(blend_frame, text="Alpha:")
lbl_alpha.pack(side="left")

alpha_entry = customtkinter.CTkEntry(blend_frame, width=50, state="disabled")
alpha_entry.pack(side="left", padx=5)

btn_apply = customtkinter.CTkButton(controls_frame, text="Apply Operation",
                                    command=apply_operation)
btn_apply.pack(side="left", padx=10)

# Result section
result_frame = customtkinter.CTkFrame(main_frame)
result_frame.pack(pady=20, fill="both", expand=True)

result_label = customtkinter.CTkLabel(result_frame, text="Result Preview")
result_label.pack(expand=True)

btn_save = customtkinter.CTkButton(result_frame, text="Save Result",
                                   command=save_result)
btn_save.pack(pady=10)

# Initial setup
update_operations_dropdown(list(OPERATION_GROUPS.keys())[0])
toggle_inputs()

root.mainloop()
