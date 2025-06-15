from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageTk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===================== Font Configuration =====================
FONT_FAMILY = "Montserrat"
HEADER_FONT = (FONT_FAMILY, 14, "bold")
SUBHEADER_FONT = (FONT_FAMILY, 12)
BODY_FONT = (FONT_FAMILY, 12)

# ===================== Image Processing Functions =====================


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

# Edge Detection Functions


def apply_prewitt(image):
    if len(image.shape) == 3:
        return np.stack([apply_prewitt(image[:, :, c]) for c in range(3)], axis=2)

    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])

    padded = np.pad(image, 1, mode='reflect')
    windows = sliding_window_view(padded, (3, 3))

    grad_x = np.sum(windows * kernel_x, axis=(2, 3))
    grad_y = np.sum(windows * kernel_y, axis=(2, 3))

    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(gradient, 0, 255).astype(np.uint8)


def apply_sobel(image):
    if len(image.shape) == 3:
        return np.stack([apply_sobel(image[:, :, c]) for c in range(3)], axis=2)

    kernel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    padded = np.pad(image, 1, mode='reflect')
    windows = sliding_window_view(padded, (3, 3))

    grad_x = np.sum(windows * kernel_x, axis=(2, 3))
    grad_y = np.sum(windows * kernel_y, axis=(2, 3))

    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.clip(gradient, 0, 255).astype(np.uint8)


def apply_laplacian(image):
    if len(image.shape) == 3:
        return np.stack([apply_laplacian(image[:, :, c]) for c in range(3)], axis=2)

    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

    padded = np.pad(image, 1, mode='reflect')
    windows = sliding_window_view(padded, (3, 3))

    laplacian = np.sum(windows * kernel, axis=(2, 3))
    return np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)


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
    },
    "Edge Detection": {
        "operations": ["Prewitt", "Sobel", "Laplacian"],
        "needs_constant": []
    },
    "Morphological": {
        "operations": [
            "Dilation",
            "Erosion",
            "Opening",
            "Closing",
            "Contour"
        ],
        "needs_constant": [
            "Dilation", "Erosion", "Opening", "Closing", "Contour"
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

        if selected_group == "Blending" and selected_operation == "Linear Blend":
            alpha_entry.configure(state="normal")

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
    original = image1_array.copy()

    if len(original.shape) == 3:
        yuv = rgb_to_yuv(original)
        y_channel = yuv[:, :, 0].copy()
        equalized_y = equalize_channel(y_channel)

        yuv[:, :, 0] = equalized_y
        equalized = yuv_to_rgb(yuv)

        original_gray = np.dot(original[..., :3], [
                               0.299, 0.587, 0.114]).astype(np.uint8)
        original_hist = np.histogram(original_gray.flatten(), 256, [0, 256])[0]
        equalized_hist = np.histogram(equalized_y.flatten(), 256, [0, 256])[0]
    else:
        equalized = equalize_channel(original)
        original_hist = np.histogram(original.flatten(), 256, [0, 256])[0]
        equalized_hist = np.histogram(equalized.flatten(), 256, [0, 256])[0]

    return equalized.astype(np.uint8), original_hist, equalized_hist


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

# Morphological operations ===============================================


def prepare_binary_image(image):
    if len(image.shape) == 3:  # Color image
        binary = binarize_image(image)
    else:
        if np.max(image) > 1:  # Grayscale image
            binary = np.where(image > 127, 255, 0).astype(np.uint8)
        else:
            binary = (image * 255).astype(np.uint8)
    return binary


def apply_morph_operation(image, kernel_size, operation):
    # Prepare binary image
    bin_img = prepare_binary_image(image)
    bin_img_01 = (bin_img // 255).astype(np.uint8)

    # Create structuring element
    pad = kernel_size // 2
    padded = np.pad(bin_img_01, pad, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))

    if operation == 'dilation':
        result = np.max(windows, axis=(2, 3))
    elif operation == 'erosion':
        result = np.min(windows, axis=(2, 3))
    else:
        raise ValueError("Invalid morphological operation")

    return (result * 255).astype(np.uint8)


def apply_dilation(image, kernel_size=3):
    return apply_morph_operation(image, kernel_size, 'dilation')


def apply_erosion(image, kernel_size=3):
    return apply_morph_operation(image, kernel_size, 'erosion')


def apply_opening(image, kernel_size=3):
    eroded = apply_erosion(image, kernel_size)
    return apply_dilation(eroded, kernel_size)


def apply_closing(image, kernel_size=3):
    dilated = apply_dilation(image, kernel_size)
    return apply_erosion(dilated, kernel_size)


def apply_contour(image, kernel_size=3):
    bin_img = prepare_binary_image(image)

    eroded = apply_erosion(bin_img, kernel_size)

    contour = np.where(bin_img > eroded, 255, 0).astype(np.uint8)
    return contour

# Validation functions ==================================================


def validate_one_image():
    if image1_array is None:
        raise ValueError("Load an image first")


def validate_two_images():
    if image1_array is None or image2_array is None:
        raise ValueError("Load both images first")
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have same dimensions")


def show_histogram_modal(original_hist, equalized_hist):
    hist_window = customtkinter.CTkToplevel()
    hist_window.title("Histogram Comparison")
    hist_window.geometry("800x500")
    hist_window.grab_set()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 4), facecolor='#2b2b2b')
    plt.subplots_adjust(wspace=0.3, left=0.1, right=0.95)

    ax1 = fig.add_subplot(121)
    ax1.bar(range(256), original_hist, width=1, color='#029cff')
    ax1.set_title("Original Histogram", color='white', fontsize=10)
    ax1.set_xlabel("Pixel Value", color='white')
    ax1.set_ylabel("Frequency", color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, color='#404040', alpha=0.3)

    ax2 = fig.add_subplot(122)
    ax2.bar(range(256), equalized_hist, width=1, color="#ff7802")
    ax2.set_title("Equalized Histogram", color='white', fontsize=10)
    ax2.set_xlabel("Pixel Value", color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, color='#404040', alpha=0.3)

    canvas = FigureCanvasTkAgg(fig, master=hist_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    btn_close = customtkinter.CTkButton(
        hist_window,
        text="Close",
        font=BODY_FONT,
        command=lambda: [plt.close('all'), hist_window.destroy()]
    )
    btn_close.pack(pady=10)

# Main operation handler ================================================


def apply_operation():
    global result_image
    selected_group = groups_dropdown.get()
    operation = operations_dropdown.get()
    constant = constant_entry.get()
    alpha_val = alpha_entry.get()

    try:
        if selected_group == "Select Category" or operation == "Select Operation":
            raise ValueError("Please select both a category and an operation")

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
                if not constant:
                    raise ValueError("Constant value is required")
                value = float(constant)
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
                if not constant:
                    raise ValueError("Threshold value is required")
                threshold = int(constant)
                result = threshold_image(threshold)
            elif operation == "Histogram Equalization":
                result, orig_hist, eq_hist = histogram_equalization()
                show_histogram_modal(orig_hist, eq_hist)

        elif selected_group == "Blending":
            validate_two_images()
            if operation == "Linear Blend":
                if not alpha_val:
                    raise ValueError("Alpha value is required")
                alpha = float(alpha_val)
                result = linear_blend(alpha)
            elif operation == "Average Images":
                result = average_images()

        elif selected_group == "Filters":
            validate_one_image()
            if not constant:
                raise ValueError("Kernel size is required")

            kernel_size = 3
            order_rank = None
            sigma = 1.0

            if operation == "Gaussian Filter":
                sigma = float(constant)
                kernel_size = min(15, 2 * int(3 * sigma) + 1)
            elif operation == "ORDER Filter":
                parts = constant.split(',')
                if len(parts) != 2:
                    raise ValueError("Use 'kernel_size,rank' format")
                kernel_size = int(parts[0])
                order_rank = int(parts[1])
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

        # ========== MORPHOLOGICAL OPERATIONS ==========
        elif selected_group == "Morphological":
            validate_one_image()
            if not constant:
                raise ValueError("Kernel size is required")
            kernel_size = int(constant)

            if operation == "Dilation":
                result = apply_dilation(image1_array, kernel_size)
            elif operation == "Erosion":
                result = apply_erosion(image1_array, kernel_size)
            elif operation == "Opening":
                result = apply_opening(image1_array, kernel_size)
            elif operation == "Closing":
                result = apply_closing(image1_array, kernel_size)
            elif operation == "Contour":
                result = apply_contour(image1_array, kernel_size)

        elif selected_group == "Edge Detection":
            validate_one_image()
            if operation == "Prewitt":
                result = apply_prewitt(image1_array)
            elif operation == "Sobel":
                result = apply_sobel(image1_array)
            elif operation == "Laplacian":
                result = apply_laplacian(image1_array)

        # Handle result display
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


# ===================== GUI Setup =====================
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title('Image Processing | by luisrefatti')
root.geometry('1280x900')

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

app_name_label = customtkinter.CTkLabel(
    header_frame,
    text="Image Processing App",
    font=HEADER_FONT,
    text_color="#029cff"
)
app_name_label.pack(side="left", padx=10)

app_author_label = customtkinter.CTkLabel(
    header_frame,
    text="Created by Luis Fernando Refatti Boff",
    font=SUBHEADER_FONT,
    text_color="white"
)
app_author_label.pack(side="left", padx=10)

# Main content
main_frame = customtkinter.CTkFrame(root)
main_frame.grid(row=1, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# Load section
load_frame = customtkinter.CTkFrame(main_frame)
load_frame.pack(pady=10, fill="x")

btn_load1 = customtkinter.CTkButton(
    load_frame,
    text="Load Image 1",
    font=BODY_FONT,
    command=lambda: load_image(1)
)
btn_load1.grid(row=0, column=0, padx=10)

btn_load2 = customtkinter.CTkButton(
    load_frame,
    text="Load Image 2",
    font=BODY_FONT,
    command=lambda: load_image(2)
)
btn_load2.grid(row=0, column=1, padx=10)

# Image previews
image_frame = customtkinter.CTkFrame(main_frame)
image_frame.pack(pady=20, fill="both", expand=True)

img1_label = customtkinter.CTkLabel(
    image_frame,
    text="Image 1 Preview",
    font=BODY_FONT
)
img1_label.grid(row=0, column=0, padx=20, sticky="nsew")

img2_label = customtkinter.CTkLabel(
    image_frame,
    text="Image 2 Preview",
    font=BODY_FONT
)
img2_label.grid(row=0, column=1, padx=20, sticky="nsew")

# Controls
controls_frame = customtkinter.CTkFrame(main_frame)
controls_frame.pack(pady=10, fill="x")

groups_dropdown = customtkinter.CTkOptionMenu(
    controls_frame,
    font=BODY_FONT,
    values=list(OPERATION_GROUPS.keys()),
    command=update_operations_dropdown
)
groups_dropdown.set("Select Category")
groups_dropdown.pack(side="left", padx=5)

operations_dropdown = customtkinter.CTkOptionMenu(
    controls_frame,
    font=BODY_FONT,
    values=[],
    command=toggle_inputs
)
operations_dropdown.set("Select Operation")
operations_dropdown.pack(side="left", padx=5)

constant_frame = customtkinter.CTkFrame(controls_frame)
constant_frame.pack(side="left", padx=5)

constant_label = customtkinter.CTkLabel(
    constant_frame,
    text="Parameter:",
    font=BODY_FONT
)
constant_label.pack(side="left")

constant_entry = customtkinter.CTkEntry(
    constant_frame,
    width=120,
    font=BODY_FONT,
    state="disabled"
)
constant_entry.pack(side="left", padx=5)

blend_frame = customtkinter.CTkFrame(controls_frame)
blend_frame.pack(side="left", padx=5)

lbl_alpha = customtkinter.CTkLabel(
    blend_frame,
    text="Alpha:",
    font=BODY_FONT
)
lbl_alpha.pack(side="left")

alpha_entry = customtkinter.CTkEntry(
    blend_frame,
    width=50,
    font=BODY_FONT,
    state="disabled"
)
alpha_entry.pack(side="left", padx=5)

btn_apply = customtkinter.CTkButton(
    controls_frame,
    text="Apply Operation",
    font=BODY_FONT,
    command=apply_operation
)
btn_apply.pack(side="left", padx=10)

# Result section
result_frame = customtkinter.CTkFrame(main_frame)
result_frame.pack(pady=20, fill="both", expand=True)

result_label = customtkinter.CTkLabel(
    result_frame,
    text="Result Preview",
    font=BODY_FONT
)
result_label.pack(expand=True)

# Save button
btn_save = customtkinter.CTkButton(
    result_frame,
    text="Save Result",
    font=BODY_FONT,
    command=save_result
)
btn_save.pack(pady=10)

# Initial setup
update_operations_dropdown(list(OPERATION_GROUPS.keys())[0])
toggle_inputs()

root.mainloop()
