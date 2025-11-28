from src.utils.CommonFunctions import *
"""
This module provides I/O utilities for the style transfer pipeline. It includes functions for:
- Loading and saving images.
- Resizing images for multi-scale processing.
- Building image pyramids.
- Handling masks and color space conversions.

Dependencies:
- OpenCV (cv2): For image resizing.
- NumPy: For numerical operations.
- scikit-image (skimage): For image I/O and color space conversions.

Ensure the following libraries are installed in your environment:
- opencv-python
- numpy
- scikit-image

To install dependencies, run:
    pip install opencv-python numpy scikit-image
"""

import os
import cv2
import numpy as np
from skimage import io as skio
from skimage.color import rgb2lab, lab2rgb

def load_image(path):
    """
    Load an image from the given path and normalize it to [0, 1].

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a float32 array in RGB format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    image = skio.imread(path)
    if image.dtype == np.uint8:
        image = image / 255.0  # Normalize to [0, 1]
    return image.astype(np.float32)

def save_image(path, image):
    """
    Save an image to the given path, converting it to uint8 format.

    Args:
        path (str): Path to save the image.
        image (np.ndarray): Image array to save (float32 in [0, 1]).
    """
    if image.dtype == np.float32:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    skio.imsave(path, image)

def resize_image(image, shape, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image to the given shape.

    Args:
        image (np.ndarray): Input image.
        shape (tuple): Desired (height, width).
        interpolation (int): Interpolation method (default: bilinear).

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)

def build_pyramid(image, num_scales):
    """
    Build an image pyramid for multi-scale processing.

    Args:
        image (np.ndarray): Input image.
        num_scales (int): Number of scales in the pyramid.

    Returns:
        list: List of images from coarse to fine scales.
    """
    pyramid = [image]
    for _ in range(1, num_scales):
        image = resize_image(image, (image.shape[0] // 2, image.shape[1] // 2))
        pyramid.append(image)
    return pyramid

def rgb_to_lab(image):
    return rgb2lab(image)

def lab_to_rgb(image):
    return lab2rgb(image)

#===============================================================================
