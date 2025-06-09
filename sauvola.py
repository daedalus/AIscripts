import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.util import img_as_float
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

def sauvola_threshold(img, window_size=15, R=128, k=0.5):
    """
    Apply Sauvola thresholding to a grayscale image.
    
    Parameters:
        img (ndarray): Input grayscale image (2D array).
        window_size (int): Size of the local neighborhood (must be odd).
        R (float): Dynamic range of standard deviation (typically 128 or 256).
        k (float): Sensitivity parameter (usually in range [0.2, 0.5]).

    Returns:
        Binary image (0 and 255).
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    img = img.astype(np.float32)
    mean = uniform_filter(img, window_size, mode='reflect')
    mean_sq = uniform_filter(img**2, window_size, mode='reflect')
    std = np.sqrt(mean_sq - mean**2)

    threshold = mean * (1 + k * ((std / R) - 1))
    binary_img = np.where(img < threshold, 0, 255).astype(np.uint8)

    return binary_img

import sys
def main():
    # Load the image
    image_path = sys.argv[1]  # Change this to your image file
    image = io.imread(image_path)

    # Convert to grayscale if it's RGB
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Convert to 0-255 range if needed
    image = img_as_ubyte(image)

    # Apply Sauvola thresholding
    binary_image = sauvola_threshold(image, window_size=15, k=0.5, R=128)

    # Display the original and thresholded image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Sauvola Thresholding")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

