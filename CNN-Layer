import math
import numpy as np
import matplotlib.pyplot as plt
import requests

from PIL import Image
from io import BytesIO


def extract_window(image: Image.Image, x: int, y: int, size: int) -> np.ndarray:
    """
    Extract a square window of width/height `size` centered at (x, y) from `image`.
    
    Parameters
    ----------
    image : PIL.Image.Image
        The original image.
    x : int
        The x-coordinate of the point at the center of the window.
    y : int
        The y-coordinate of the point at the center of the window.
    size : int
        The width and height of the window.

    Returns
    -------
    np.ndarray or None
        A NumPy array representing the cropped window if valid; otherwise None.
    """
    if size < 0:
        return None
    
    img_array = np.array(image)
    height, width = img_array.shape[0], img_array.shape[1]

    if size >= width or size >= height:
        # If the requested window size is bigger than the image, return the entire image.
        return img_array

    left = x - math.floor(size / 2)
    top = y - math.ceil(size / 2)
    right = left + size
    bottom = top + size

    # Check for out-of-bounds
    if left < 0 or top < 0 or right > width or bottom > height:
        return None

    # Slice the sub-array
    return img_array[top:bottom, left:right]


def conv_window(window: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a window with a 2D kernel (normal convolution, no dilation).
    
    Parameters
    ----------
    window : np.ndarray
        The 2D image patch to convolve.
    kernel : np.ndarray
        The 2D kernel.
    
    Returns
    -------
    np.ndarray
        Result of the convolution operation.
    """
    matrix = np.array(window)
    kernel_size = len(kernel)
    
    out_y = matrix.shape[0] - kernel_size + 1
    out_x = matrix.shape[1] - kernel_size + 1
    result = np.zeros((out_y, out_x), dtype=np.float32)

    for i in range(out_y):
        for j in range(out_x):
            patch = matrix[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(patch * kernel)

    return result


def dilation_conv_window(window: np.ndarray, kernel: np.ndarray, dilation_factor: int) -> np.ndarray:
    """
    Perform a dilated convolution on a window with the given kernel and dilation factor.
    
    Parameters
    ----------
    window : np.ndarray
        The 2D image patch to be convolved.
    kernel : np.ndarray
        The 2D convolution kernel.
    dilation_factor : int
        The factor by which to dilate the kernel.
    
    Returns
    -------
    np.ndarray
        The dilated convolution result.
    """
    matrix = np.array(window)
    kernel_size = len(kernel)

    # Effective size of the dilated kernel
    inner_matrix_size = kernel_size + (kernel_size - 1) * (dilation_factor - 1)

    out_y = matrix.shape[0] - inner_matrix_size + 1
    out_x = matrix.shape[1] - inner_matrix_size + 1

    result = np.zeros((out_y, out_x), dtype=np.float32)

    # Scan through the area where the dilated kernel can fit
    for i in range(out_y):
        for j in range(out_x):
            temp_sum = 0.0
            # Summation over the dilated positions
            for m in range(kernel_size):
                for n in range(kernel_size):
                    y_coord = i + m * dilation_factor
                    x_coord = j + n * dilation_factor
                    temp_sum += kernel[m, n] * matrix[y_coord, x_coord]
            result[i, j] = temp_sum

    return result


def zero_pad_image(image: Image.Image, pad_size_left_bottom: int, pad_size_top_right: int) -> np.ndarray:
    """
    Zero-pad an image along all sides by specified amounts.
    
    Parameters
    ----------
    image : PIL.Image.Image
        The original image.
    pad_size_left_bottom : int
        Number of zero-padding pixels to add on the left and bottom sides.
    pad_size_top_right : int
        Number of zero-padding pixels to add on the right and top sides.
    
    Returns
    -------
    np.ndarray
        The padded 2D image array.
    """
    img_array = np.array(image)
    original_size_y, original_size_x = img_array.shape[:2]

    # New dimensions after padding
    new_size_y = original_size_y + pad_size_left_bottom + pad_size_top_right
    new_size_x = original_size_x + pad_size_left_bottom + pad_size_top_right

    pad_img = np.zeros((new_size_y, new_size_x), dtype=img_array.dtype)

    # Insert original image into padded area
    pad_img[pad_size_top_right:pad_size_top_right + original_size_y,
            pad_size_left_bottom:pad_size_left_bottom + original_size_x] = img_array

    return pad_img


def get_windows(image: Image.Image, window_size: int):
    """
    Yield all square windows of size `window_size` centered on every valid pixel in `image`.
    
    Parameters
    ----------
    image : PIL.Image.Image
        The image from which to extract windows.
    window_size : int
        The height and width of each window.
    
    Yields
    ------
    (x_coord, y_coord, window)
        - x_coord, y_coord: The center pixel in the original (unpadded) image.
        - window: The extracted window (patch) as a NumPy array.
    """
    original_size_x, original_size_y = image.size
    pad_left_bottom = window_size // 2
    pad_top_right = window_size - pad_left_bottom

    padded_image = zero_pad_image(image, pad_left_bottom, pad_top_right)

    for y in range(original_size_y):
        for x in range(original_size_x):
            window = extract_window(
                padded_image,
                x + pad_left_bottom,
                y + pad_top_right,
                window_size
            )
            yield (x, y, window)


def convolve_dilation(image: Image.Image, kernel: np.ndarray, dilation: int) -> np.ndarray:
    """
    Return a convolved image given an image and a kernel, supporting a dilation factor.
    
    Parameters
    ----------
    image : PIL.Image.Image
        The image to be convolved (in grayscale).
    kernel : np.ndarray
        The 2D convolution kernel.
    dilation : int
        Dilation factor (1 = normal convolution; >1 = dilated).
    
    Returns
    -------
    np.ndarray
        The 2D array of the convolved image.
    """
    # Initialize an output array the same size as the input image
    convolved_image = np.zeros((image.size[1], image.size[0]), dtype=np.float32)

    # For each window in the image, compute the (dilated) convolution
    for x_coord, y_coord, window in get_windows(image, 5):
        if window is not None:
            conv_res = dilation_conv_window(window, kernel, dilation)
            # We expect conv_res to be a single value if kernel == window size,
            # or a small patch if not. We’ll store the top-left in convolved_image.
            # For simplicity, we assume a 1x1 result if window size == kernel size
            convolved_image[y_coord, x_coord] = conv_res[0, 0]
    
    return convolved_image


# ------------------------------------------------------------------------
#                            Testing / Usage
# ------------------------------------------------------------------------

def main():
    """
    Simple demonstration of the hard-coded CNN layer operations.
    """
    # Download a test image
    url = "https://dl.sphericalcow.xyz/ecse552/conv_hmwk/mandrill.tiff"
    r = requests.get(url)
    test_image = Image.open(BytesIO(r.content)).convert('L')  # convert to grayscale

    # Example kernel (Sobel X)
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)

    # Perform a dilated convolution
    dilation_factor = 2
    c_img = convolve_dilation(test_image, kernel, dilation_factor)

    # Display the convolved result
    plt.imshow(c_img, cmap='gray')
    plt.title(f"Dilated Convolution (factor={dilation_factor})")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
