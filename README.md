# proc-img
A Python-based image processing application using CustomTkinter for performing arithmetic operations on images.

This project was developed for the **Image Processing** course by Professor **Daniel Menin** at the **Universidade Regional Integrada do Alto Uruguai e das Missões (URI Erechim)**.

---

## Overview

This application provides a graphical user interface (GUI) for performing various image processing operations.  
It leverages:

- `tkinter` for the GUI  
- `Pillow (PIL)` for image handling  
- `customtkinter` for a modern look and feel  

The core image processing functionalities are implemented manually, providing a deeper understanding of the underlying algorithms.

---

## Features

The application supports a wide range of image processing operations, categorized as follows:

### Arithmetic Operations

- **Add Images**: Sums pixel values of two images  
- **Subtract Images**: Subtracts pixel values of one image from another  
- **Add Constant**: Adds a constant value to each pixel  
- **Subtract Constant**: Subtracts a constant value from each pixel  
- **Multiply by Constant**: Multiplies each pixel value by a constant  
- **Divide by Constant**: Divides each pixel value by a constant  

### Logic Operations

- **AND**: Bitwise AND operation on two images  
- **OR**: Bitwise OR operation on two images  
- **XOR**: Bitwise XOR operation on two images  
- **NOT**: Inverts the pixel values of an image  

### Geometric Operations

- **Flip Horizontal**: Flips an image horizontally  
- **Flip Vertical**: Flips an image vertically  

### Color Operations

- **RGB to Grayscale**: Converts an RGB image to grayscale  
- **Thresholding**: Applies a binary threshold to an image  
- **Histogram Equalization**: Enhances contrast by equalizing the histogram and show the result of the original histogram and the equalized histogram

### Blending Operations

- **Linear Blend**: Blends two images with a specified alpha  
- **Average Images**: Averages pixel values  

### Filters

- **MAX Filter**: Applies a dilation (max filter)  
- **MIN Filter**: Applies an erosion (min filter)  
- **MEAN Filter**: Smooths the image with an average filter  
- **MEDIAN Filter**: Reduces noise with a median filter (good for SP noise)
- **ORDER Filter**: Applies a rank filter (order-statistic)  
- **Conservative Smooth**: Preserves edges while smoothing  
- **Gaussian Filter**: Applies Gaussian blur  

### Edge Detection

- **Prewitt**: Edge detection with the Prewitt operator  
- **Sobel**: Edge detection with the Sobel operator  
- **Laplacian**: Edge detection with the Laplacian operator  

### Morphological Operations

- **Dilation**: Expands bright regions  
- **Erosion**: Shrinks bright regions  
- **Opening**: Erosion followed by dilation  
- **Closing**: Dilation followed by erosion  
- **Contour**: Extracts object contours  

---

## Technologies Used

- **Python 3**
- `tkinter` – GUI toolkit  
- `customtkinter` – Modern tkinter widgets  
- `Pillow (PIL)` – Image manipulation  
- `matplotlib` – Histogram plotting
- `pyinstaller` – Running the app 

---

## Usage

Open the file 'app.exe' on the 'dist' folder.

### 1. Load images

- Click **"Load Image 1"** to load the primary image.  
- Click **"Load Image 2"** if the selected operation requires two images (e.g., *Add Images*, *Linear Blend*, *AND*).  

### 2. Select operation

- Choose a **Category** from the first dropdown (e.g., *Arithmetic*, *Color*, *Filters*).  
- Select an **Operation** from the second dropdown.  

### 3. Enter constant value (if required)

- If the selected operation needs a value (e.g., *Add Constant*, *Thresholding*), a field will become active.  
- Enter the desired number.  

### 4. Apply operation

- Click the **"Apply Operation"** button.  

### 5. View result

- The processed image will be shown in the **"Result Image"** panel.  
- For **Histogram Equalization**, both original and equalized histograms will be displayed.  

### 6. Save result

- Click **"Save Result"** to download the processed image.  

---

## License

This project was developed for educational purposes.  
Feel free to use, modify, and distribute it with proper attribution.