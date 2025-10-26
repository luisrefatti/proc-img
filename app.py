# recreating photoshop

from tkinter import filedialog, messagebox
import customtkinter
from PIL import Image, ImageTk
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===================== FONT CONFIGURATION =====================
FONT_FAMILY = "Montserrat"
HEADER_FONT = (FONT_FAMILY, 14, "bold")
SUBHEADER_FONT = (FONT_FAMILY, 12)
BODY_FONT = (FONT_FAMILY, 12)

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

# ===================== MATH UTILITY FUNCTIONS =====================


def clip(value, min_val, max_val):
    return min(max(value, min_val), max_val)


def create_2d_array(width, height, default=0):
    return [[default for _ in range(width)] for _ in range(height)]


def create_3d_array(width, height, depth=3, default=0):
    return [[[default for _ in range(depth)] for _ in range(width)] for _ in range(height)]


def get_pixel(image, x, y):
    height = len(image)
    width = len(image[0])
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return image[y][x]


def apply_kernel(image, kernel):
    height = len(image)
    width = len(image[0])
    k_height = len(kernel)
    k_width = len(kernel[0])
    k_half_h = k_height // 2
    k_half_w = k_width // 2

    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            accum = 0.0
            for ky in range(k_height):
                for kx in range(k_width):
                    px = x + kx - k_half_w
                    py = y + ky - k_half_h

                    if px < 0:
                        px = -px
                    if px >= width:
                        px = 2 * width - px - 1
                    if py < 0:
                        py = -py
                    if py >= height:
                        py = 2 * height - py - 1

                    pixel = image[py][px]
                    if isinstance(pixel, tuple):
                        pixel = 0.299 * pixel[0] + 0.587 * \
                            pixel[1] + 0.114 * pixel[2]

                    accum += pixel * kernel[ky][kx]

            result[y][x] = clip(int(accum), 0, 255)

    return result


def sliding_window_view(image, kernel_size):
    height = len(image)
    width = len(image[0])
    k_half = kernel_size // 2
    windows = []

    for y in range(height):
        row_windows = []
        for x in range(width):
            window = []
            for ky in range(-k_half, k_half + 1):
                for kx in range(-k_half, k_half + 1):
                    px = x + kx
                    py = y + ky

                    if px < 0:
                        px = -px
                    if px >= width:
                        px = 2 * width - px - 1
                    if py < 0:
                        py = -py
                    if py >= height:
                        py = 2 * height - py - 1

                    window.append(image[py][px])
            row_windows.append(window)
        windows.append(row_windows)

    return windows

# ===================== IMAGE PROCESSING FUNCTIONS =====================


def create_gaussian_kernel(size, sigma):
    kernel = [[0.0] * size for _ in range(size)]
    center = size // 2
    sum_val = 0.0

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            value = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i][j] = value
            sum_val += value

    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_val

    return kernel


def apply_gaussian_filter_manual(image, sigma=1.0, kernel_size=5):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return apply_kernel(image, kernel)

# --------------------- Edge Detection ---------------------


def apply_prewitt(image):
    if isinstance(image[0][0], tuple):
        gray_image = rgb_to_grayscale(image)
    else:
        gray_image = image

    kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    kernel_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    grad_x = apply_kernel(gray_image, kernel_x)
    grad_y = apply_kernel(gray_image, kernel_y)

    height = len(gray_image)
    width = len(gray_image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            gradient = math.sqrt(grad_x[y][x]**2 + grad_y[y][x]**2)
            scaled_gradient = min(255, gradient * 3)
            result[y][x] = clip(int(scaled_gradient), 0, 255)

    return result


def apply_sobel(image):
    if isinstance(image[0][0], tuple):
        gray_image = rgb_to_grayscale(image)
    else:
        gray_image = image

    kernel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    grad_x = apply_kernel(gray_image, kernel_x)
    grad_y = apply_kernel(gray_image, kernel_y)

    height = len(gray_image)
    width = len(gray_image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            gradient = math.sqrt(grad_x[y][x]**2 + grad_y[y][x]**2)
            scaled_gradient = min(255, gradient * 3)
            result[y][x] = clip(int(scaled_gradient), 0, 255)

    return result


def apply_laplacian(image):
    if isinstance(image[0][0], tuple):
        gray_image = rgb_to_grayscale(image)
    else:
        gray_image = image

    kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian = apply_kernel(gray_image, kernel)

    height = len(gray_image)
    width = len(gray_image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            result[y][x] = clip(abs(laplacian[y][x]) * 2, 0, 255)

    return result

# --------------------- Arithmetic Operations ---------------------


def add_images(img1, img2):
    height = len(img1)
    width = len(img1[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            if isinstance(img1[y][x], tuple) and isinstance(img2[y][x], tuple):
                r = clip(img1[y][x][0] + img2[y][x][0], 0, 255)
                g = clip(img1[y][x][1] + img2[y][x][1], 0, 255)
                b = clip(img1[y][x][2] + img2[y][x][2], 0, 255)
                result[y][x] = (r, g, b)
            else:
                result[y][x] = clip(img1[y][x] + img2[y][x], 0, 255)

    return result


def subtract_images(img1, img2):
    height = len(img1)
    width = len(img1[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            if isinstance(img1[y][x], tuple) and isinstance(img2[y][x], tuple):
                r = clip(img1[y][x][0] - img2[y][x][0], 0, 255)
                g = clip(img1[y][x][1] - img2[y][x][1], 0, 255)
                b = clip(img1[y][x][2] - img2[y][x][2], 0, 255)
                result[y][x] = (r, g, b)
            else:
                result[y][x] = clip(img1[y][x] - img2[y][x], 0, 255)

    return result


def apply_constant_operation(image, operation, value):
    height = len(image)
    width = len(image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            if isinstance(image[y][x], tuple):
                r, g, b = image[y][x]
                if operation == 'add':
                    r = clip(r + value, 0, 255)
                    g = clip(g + value, 0, 255)
                    b = clip(b + value, 0, 255)
                elif operation == 'subtract':
                    r = clip(r - value, 0, 255)
                    g = clip(g - value, 0, 255)
                    b = clip(b - value, 0, 255)
                elif operation == 'multiply':
                    r = clip(r * value, 0, 255)
                    g = clip(g * value, 0, 255)
                    b = clip(b * value, 0, 255)
                elif operation == 'divide':
                    r = clip(r / value, 0, 255)
                    g = clip(g / value, 0, 255)
                    b = clip(b / value, 0, 255)
                result[y][x] = (int(r), int(g), int(b))
            else:
                if operation == 'add':
                    result[y][x] = clip(image[y][x] + value, 0, 255)
                elif operation == 'subtract':
                    result[y][x] = clip(image[y][x] - value, 0, 255)
                elif operation == 'multiply':
                    result[y][x] = clip(image[y][x] * value, 0, 255)
                elif operation == 'divide':
                    result[y][x] = clip(image[y][x] / value, 0, 255)

    return result

# --------------------- Geometric Operations ---------------------


def flip_horizontal(image):
    height = len(image)
    width = len(image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            result[y][x] = image[y][width - x - 1]

    return result


def flip_vertical(image):
    height = len(image)
    width = len(image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            result[y][x] = image[height - y - 1][x]

    return result

# --------------------- Color Operations ---------------------


def rgb_to_grayscale(image):
    height = len(image)
    width = len(image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            r, g, b = image[y][x]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            result[y][x] = clip(int(gray), 0, 255)

    return result


def threshold_image(image, threshold):
    height = len(image)
    width = len(image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            if isinstance(image[y][x], tuple):
                r, g, b = image[y][x]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                result[y][x] = 255 if gray > threshold else 0
            else:
                result[y][x] = 255 if image[y][x] > threshold else 0

    return result

# --------------------- Blending Operations ---------------------


def linear_blend(img1, img2, alpha):
    height = len(img1)
    width = len(img1[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            if isinstance(img1[y][x], tuple) and isinstance(img2[y][x], tuple):
                r1, g1, b1 = img1[y][x]
                r2, g2, b2 = img2[y][x]
                r = alpha * r1 + (1 - alpha) * r2
                g = alpha * g1 + (1 - alpha) * g2
                b = alpha * b1 + (1 - alpha) * b2
                result[y][x] = (int(r), int(g), int(b))
            else:
                result[y][x] = alpha * img1[y][x] + (1 - alpha) * img2[y][x]

    return result


def average_images(img1, img2):
    return linear_blend(img1, img2, 0.5)

# --------------------- Logical Operations ---------------------


def logical_operation(img1, img2, operation):
    height = len(img1)
    width = len(img1[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            if isinstance(img1[y][x], tuple):
                r, g, b = img1[y][x]
                val1 = 1 if (0.299 * r + 0.587 * g + 0.114 * b) > 127 else 0
            else:
                val1 = 1 if img1[y][x] > 127 else 0

            if operation != "NOT":
                if isinstance(img2[y][x], tuple):
                    r, g, b = img2[y][x]
                    val2 = 1 if (0.299 * r + 0.587 * g +
                                 0.114 * b) > 127 else 0
                else:
                    val2 = 1 if img2[y][x] > 127 else 0

            if operation == "AND":
                res = min(val1, val2)
            elif operation == "OR":
                res = max(val1, val2)
            elif operation == "XOR":
                res = (val1 + val2) % 2
            elif operation == "NOT":
                res = 1 - val1

            result[y][x] = res * 255

    return result

# ===================== HISTOGRAM OPERATIONS =====================


def compute_histogram(image):
    hist = [0] * 256
    height = len(image)
    width = len(image[0])

    for y in range(height):
        for x in range(width):
            val = image[y][x]
            if 0 <= val < 256:
                hist[val] += 1
    return hist


def histogram_equalization(image):
    if isinstance(image[0][0], tuple):
        gray_img = rgb_to_grayscale(image)
        orig_hist = compute_histogram(gray_img)

        yuv = rgb_to_yuv(image)
        height = len(image)
        width = len(image[0])

        y_channel = create_2d_array(width, height, 0.0)
        for y in range(height):
            for x in range(width):
                y_channel[y][x] = yuv[y][x][0]

        eq_y, cdf_normalized = equalize_channel(y_channel)

        new_yuv = create_3d_array(width, height, 3, 0.0)
        for y in range(height):
            for x in range(width):
                new_yuv[y][x] = (eq_y[y][x], yuv[y][x][1], yuv[y][x][2])

        equalized = yuv_to_rgb(new_yuv)

        eq_y_int = create_2d_array(width, height, 0)
        for y in range(height):
            for x in range(width):
                eq_y_int[y][x] = int(eq_y[y][x])
        eq_hist = compute_histogram(eq_y_int)

        return equalized, orig_hist, eq_hist
    else:
        orig_hist = compute_histogram(image)
        equalized, cdf_normalized = equalize_channel(image)
        eq_hist = compute_histogram(equalized)
        return equalized, orig_hist, eq_hist


def equalize_channel(channel):
    height = len(channel)
    width = len(channel[0])
    int_channel = create_2d_array(width, height, 0)
    for y in range(height):
        for x in range(width):
            int_val = int(round(channel[y][x]))
            int_channel[y][x] = clip(int_val, 0, 255)

    hist = compute_histogram(int_channel)
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]

    cdf_min = next((val for val in cdf if val > 0), 0)
    cdf_max = cdf[-1]

    cdf_normalized = [0] * 256
    if cdf_max - cdf_min > 0:
        for i in range(256):
            cdf_normalized[i] = int(
                255 * (cdf[i] - cdf_min) / (cdf_max - cdf_min))
    else:
        cdf_normalized = cdf

    equalized = create_2d_array(width, height, 0)
    for y in range(height):
        for x in range(width):
            val = int_channel[y][x]
            equalized[y][x] = cdf_normalized[val]

    return equalized, cdf_normalized

# ===================== COLOR CONVERSION FUNCTIONS =====================


def rgb_to_yuv(rgb_image):
    height = len(rgb_image)
    width = len(rgb_image[0])
    yuv = create_3d_array(width, height, 3, 0.0)

    for y in range(height):
        for x in range(width):
            r, g, b = rgb_image[y][x]
            y_val = 0.299 * r + 0.587 * g + 0.114 * b
            u_val = -0.147 * r - 0.289 * g + 0.436 * b
            v_val = 0.615 * r - 0.515 * g - 0.100 * b
            yuv[y][x] = (y_val, u_val, v_val)

    return yuv


def yuv_to_rgb(yuv_image):
    height = len(yuv_image)
    width = len(yuv_image[0])
    rgb = create_3d_array(width, height, 3, 0)

    for y in range(height):
        for x in range(width):
            y_val, u_val, v_val = yuv_image[y][x]
            r = y_val + 1.140 * v_val
            g = y_val - 0.395 * u_val - 0.581 * v_val
            b = y_val + 2.032 * u_val

            r = clip(r, 0, 255)
            g = clip(g, 0, 255)
            b = clip(b, 0, 255)

            rgb[y][x] = (int(r), int(g), int(b))

    return rgb

# --------------------- Spatial Filters ---------------------


def apply_filter(channel, kernel_size, filter_type, order_rank=None):
    height = len(channel)
    width = len(channel[0])
    k_half = kernel_size // 2
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            window = []
            for ky in range(-k_half, k_half + 1):
                for kx in range(-k_half, k_half + 1):
                    px = x + kx
                    py = y + ky

                    if px < 0:
                        px = -px
                    if px >= width:
                        px = 2 * width - px - 1
                    if py < 0:
                        py = -py
                    if py >= height:
                        py = 2 * height - py - 1

                    pixel = channel[py][px]
                    if isinstance(pixel, tuple):
                        pixel = 0.299 * pixel[0] + 0.587 * \
                            pixel[1] + 0.114 * pixel[2]
                    window.append(pixel)

            if filter_type == 'max':
                result[y][x] = max(window)
            elif filter_type == 'min':
                result[y][x] = min(window)
            elif filter_type == 'mean':
                result[y][x] = int(sum(window) / len(window))
            elif filter_type == 'median':
                window.sort()
                result[y][x] = window[len(window)//2]
            elif filter_type == 'order':
                window.sort()
                rank = min(max(order_rank, 0), len(window)-1)
                result[y][x] = window[rank]
            elif filter_type == 'conservative':
                min_val = min(window)
                max_val = max(window)
                center = window[len(window)//2]
                min_diff = abs(center - min_val)
                max_diff = abs(center - max_val)
                result[y][x] = min_val if min_diff < max_diff else max_val

    return result

# --------------------- Morphological Operations ---------------------


def apply_morph_operation(image, kernel_size, operation):
    if isinstance(image[0][0], tuple):
        gray_image = rgb_to_grayscale(image)
    else:
        gray_image = image

    height = len(gray_image)
    width = len(gray_image[0])
    k_half = kernel_size // 2
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            window = []
            for ky in range(-k_half, k_half + 1):
                for kx in range(-k_half, k_half + 1):
                    px = x + kx
                    py = y + ky

                    if px < 0:
                        px = -px
                    if px >= width:
                        px = 2 * width - px - 1
                    if py < 0:
                        py = -py
                    if py >= height:
                        py = 2 * height - py - 1

                    pixel = gray_image[py][px]
                    window.append(pixel)

            if operation == 'dilation':
                result[y][x] = max(window)
            elif operation == 'erosion':
                result[y][x] = min(window)

    return result


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
    if isinstance(image[0][0], tuple):
        gray_image = rgb_to_grayscale(image)
    else:
        gray_image = image

    dilated = apply_dilation(gray_image, kernel_size)
    eroded = apply_erosion(gray_image, kernel_size)

    height = len(gray_image)
    width = len(gray_image[0])
    result = create_2d_array(width, height, 0)

    for y in range(height):
        for x in range(width):
            contour = dilated[y][x] - eroded[y][x]
            result[y][x] = clip(abs(contour), 0, 255)

    return result

# ===================== HELPER FUNCTIONS =====================


def validate_one_image():
    if image1_array is None:
        raise ValueError("Load an image first")


def validate_two_images():
    if image1_array is None or image2_array is None:
        raise ValueError("Load both images first")
    if len(image1_array) != len(image2_array) or len(image1_array[0]) != len(image2_array[0]):
        raise ValueError("Images must have same dimensions")

# ===================== GUI FUNCTIONS =====================


def load_image(img_num):
    global image1, image2, image1_array, image2_array
    file_path = filedialog.askopenfilename(filetypes=[
        ("Image Files", "*.bmp *.jpg *.png *.jpeg *.tif")
    ])

    if not file_path:
        return

    try:
        img = Image.open(file_path).convert("RGB")
        width, height = img.size
        pixels = list(img.getdata())

        img_array = []
        for y in range(height):
            row = []
            for x in range(width):
                r, g, b = pixels[y * width + x]
                row.append((r, g, b))
            img_array.append(row)

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

# --------------------- Operation Controls ---------------------


def update_operations_dropdown(selected_group):
    operations = OPERATION_GROUPS[selected_group]["operations"]
    operations_dropdown.configure(values=operations)
    operations_dropdown.set(operations[0] if operations else "")
    toggle_inputs()


def toggle_inputs(event=None):
    selected_group = groups_dropdown.get()
    selected_operation = operations_dropdown.get()

    constant_entry.configure(state="disabled")
    constant_entry.delete(0, 'end')

    if selected_group in OPERATION_GROUPS:
        group_data = OPERATION_GROUPS[selected_group]

        if selected_operation in group_data["needs_constant"]:
            constant_entry.configure(state="normal")

# --------------------- Histogram Visualization ---------------------


def update_histogram_plots(original_hist, equalized_hist):
    hist_ax1.clear()
    hist_ax2.clear()

    hist_ax1.bar(range(256), original_hist, width=1, color='#029cff')
    hist_ax1.set_title("Original Histogram", color='white', fontsize=10)
    hist_ax1.set_xlabel("Pixel Value", color='white')
    hist_ax1.set_ylabel("Frequency", color='white')
    hist_ax1.tick_params(colors='white')
    hist_ax1.grid(True, color='#404040', alpha=0.3)

    hist_ax2.bar(range(256), equalized_hist, width=1, color="#ff7802")
    hist_ax2.set_title("Equalized Histogram", color='white', fontsize=10)
    hist_ax2.set_xlabel("Pixel Value", color='white')
    hist_ax2.tick_params(colors='white')
    hist_ax2.grid(True, color='#404040', alpha=0.3)

    hist_canvas.draw()

# --------------------- Operation Handler ---------------------


def apply_operation():
    global result_image
    selected_group = groups_dropdown.get()
    operation = operations_dropdown.get()
    constant = constant_entry.get()

    try:
        histogram_frame.grid_remove()

        if selected_group == "Select Category" or operation == "Select Operation":
            raise ValueError("Please select both a category and an operation")

        if selected_group == "Arithmetic":
            if "Images" in operation:
                validate_two_images()
                if operation == "Add Images":
                    result = add_images(image1_array, image2_array)
                elif operation == "Subtract Images":
                    result = subtract_images(image1_array, image2_array)
            else:
                validate_one_image()
                op_type = operation.split()[0].lower()
                if not constant:
                    raise ValueError("Constant value is required")
                value = float(constant)
                result = apply_constant_operation(image1_array, op_type, value)

        elif selected_group == "Logic":
            validate_one_image()
            if operation != "NOT":
                validate_two_images()
                result = logical_operation(
                    image1_array, image2_array, operation)
            else:
                result = logical_operation(image1_array, None, operation)

        elif selected_group == "Geometric":
            validate_one_image()
            if operation == "Flip Horizontal":
                result = flip_horizontal(image1_array)
            elif operation == "Flip Vertical":
                result = flip_vertical(image1_array)

        elif selected_group == "Color":
            validate_one_image()
            if operation == "RGB to Grayscale":
                result = rgb_to_grayscale(image1_array)
            elif operation == "Thresholding":
                if not constant:
                    raise ValueError("Threshold value is required")
                threshold = int(constant)
                result = threshold_image(image1_array, threshold)
            elif operation == "Histogram Equalization":
                result, orig_hist, eq_hist = histogram_equalization(
                    image1_array)
                histogram_frame.grid()
                update_histogram_plots(orig_hist, eq_hist)

        elif selected_group == "Blending":
            validate_two_images()
            if operation == "Linear Blend":
                if not constant:
                    raise ValueError("Alpha value is required")
                alpha = float(constant)
                result = linear_blend(image1_array, image2_array, alpha)
            elif operation == "Average Images":
                result = average_images(image1_array, image2_array)

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
                result = apply_gaussian_filter_manual(
                    image1_array, sigma, kernel_size)
            elif operation == "ORDER Filter":
                parts = constant.split(',')
                if len(parts) != 2:
                    raise ValueError("Use 'kernel_size,rank' format")
                kernel_size = int(parts[0])
                order_rank = int(parts[1])

                if isinstance(image1_array[0][0], tuple):
                    height = len(image1_array)
                    width = len(image1_array[0])
                    result = create_3d_array(width, height, 3, 0)
                    for c in range(3):
                        channel = []
                        for y in range(height):
                            row = []
                            for x in range(width):
                                row.append(image1_array[y][x][c])
                            channel.append(row)

                        filtered = apply_filter(
                            channel, kernel_size, 'order', order_rank)

                        for y in range(height):
                            for x in range(width):
                                result[y][x][c] = filtered[y][x]
                else:
                    result = apply_filter(
                        image1_array, kernel_size, 'order', order_rank)
            else:
                kernel_size = int(constant)
                filter_type = operation.split()[0].lower()

                if isinstance(image1_array[0][0], tuple):
                    height = len(image1_array)
                    width = len(image1_array[0])
                    result = create_3d_array(width, height, 3, 0)
                    for c in range(3):
                        channel = []
                        for y in range(height):
                            row = []
                            for x in range(width):
                                row.append(image1_array[y][x][c])
                            channel.append(row)

                        filtered = apply_filter(
                            channel, kernel_size, filter_type)

                        for y in range(height):
                            for x in range(width):
                                result[y][x][c] = filtered[y][x]
                else:
                    result = apply_filter(
                        image1_array, kernel_size, filter_type)

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

        height = len(result)
        width = len(result[0])

        pixels = []
        for y in range(height):
            for x in range(width):
                pixel = result[y][x]
                if isinstance(pixel, tuple):
                    pixels.append(pixel)
                elif isinstance(pixel, list) and len(pixel) == 3:
                    pixels.append(tuple(pixel))
                else:
                    pixels.append((pixel, pixel, pixel))

        img = Image.new('RGB', (width, height))
        img.putdata(pixels)
        result_image = img

        display_image(img, result_label)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ===================== GUI SETUP =====================
image1 = None
image2 = None
result_image = None
image1_array = None
image2_array = None

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title('Image Processing | by luisrefatti')
root.geometry('1280x900')
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Header section
header_frame = customtkinter.CTkFrame(root)
header_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

app_name_label = customtkinter.CTkLabel(
    header_frame,
    text="Image Processing App",
    font=HEADER_FONT,
    text_color="#029cff"
)
app_name_label.pack(side="left", padx=10)

app_name_label = customtkinter.CTkLabel(
    header_frame,
    text="|",
    font=HEADER_FONT,
    text_color="#7d7d7d"
)
app_name_label.pack(side="left", padx=10)

app_author_label = customtkinter.CTkLabel(
    header_frame,
    text="Created by Luis Fernando Refatti Boff",
    font=SUBHEADER_FONT,
    text_color="white"
)
app_author_label.pack(side="left", padx=10)

# Main content frame
main_frame = customtkinter.CTkFrame(root)
main_frame.grid(row=1, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

# --------------------- Top Section: Image Loading and Histograms ---------------------
top_frame = customtkinter.CTkFrame(main_frame)
top_frame.pack(fill='x', padx=10, pady=10)

load_frame = customtkinter.CTkFrame(top_frame)
load_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

btn_load1 = customtkinter.CTkButton(
    load_frame,
    text="Load Image 1",
    font=BODY_FONT,
    command=lambda: load_image(1)
)
btn_load1.pack(pady=5)

btn_load2 = customtkinter.CTkButton(
    load_frame,
    text="Load Image 2",
    font=BODY_FONT,
    command=lambda: load_image(2)
)
btn_load2.pack(pady=5)

image_frame = customtkinter.CTkFrame(top_frame)
image_frame.grid(row=0, column=1, padx=10, pady=10)

img1_label = customtkinter.CTkLabel(
    image_frame,
    text="Image 1 Preview",
    font=BODY_FONT
)
img1_label.grid(row=0, column=0, padx=20, pady=5, sticky="nsew")

img2_label = customtkinter.CTkLabel(
    image_frame,
    text="Image 2 Preview",
    font=BODY_FONT
)
img2_label.grid(row=0, column=1, padx=20, pady=5, sticky="nsew")

histogram_frame = customtkinter.CTkFrame(top_frame)
histogram_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
histogram_frame.grid_remove()

plt.style.use('dark_background')
hist_fig = plt.Figure(figsize=(6, 3), facecolor='#2b2b2b')
hist_fig.subplots_adjust(wspace=0.3, left=0.1, right=0.95)

hist_ax1 = hist_fig.add_subplot(121)
hist_ax2 = hist_fig.add_subplot(122)

hist_canvas = FigureCanvasTkAgg(hist_fig, master=histogram_frame)
hist_canvas.draw()
hist_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

# --------------------- Operation Controls Section ---------------------
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

btn_apply = customtkinter.CTkButton(
    controls_frame,
    text="Apply Operation",
    font=BODY_FONT,
    command=apply_operation
)
btn_apply.pack(side="left", padx=10)

# --------------------- Result Section ---------------------
result_frame = customtkinter.CTkFrame(main_frame)
result_frame.pack(pady=20, fill="both", expand=True)

result_label = customtkinter.CTkLabel(
    result_frame,
    text="Result Preview",
    font=BODY_FONT
)
result_label.pack(expand=True)

btn_save = customtkinter.CTkButton(
    result_frame,
    text="Save Result",
    font=BODY_FONT,
    command=save_result
)
btn_save.pack(pady=10)

# ===================== APPLICATION INITIALIZATION =====================
update_operations_dropdown(list(OPERATION_GROUPS.keys())[0])
toggle_inputs()
root.mainloop()
