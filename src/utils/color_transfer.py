"""
This module provides utilities for color transfer in the style transfer pipeline. It includes functions for:
- Matching color statistics between content and style images.
- Applying per-region color transfer based on segmentation masks.

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

import sys
import os
from skimage.exposure import match_histograms
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from skimage.color import rgb2lab, lab2rgb
from src.utils.io import load_image, save_image, resize_image, rgb_to_lab, lab_to_rgb


def match_color_stats(content, style, mask=None, strength=1.0):
    """
    Improved histogram-based color transfer (stronger and richer).
    Works in LAB color space and supports masked regions.

    Args:
        content (np.ndarray): RGB content image.
        style (np.ndarray): RGB style image.
        mask (np.ndarray, optional): Region mask. If None → global.
        strength (float): 0–1.0 blend between original and transferred.

    Returns:
        np.ndarray: Color-transferred RGB image.
    """

    # Convert to LAB (better separation of luminance/chrominance)
    content_lab = rgb_to_lab(content)
    style_lab = rgb_to_lab(style)

    # No mask → global histogram matching
    if mask is None:
        transferred = np.zeros_like(content_lab)
        for c in range(3):
            transferred[..., c] = match_histograms(
                content_lab[..., c], style_lab[..., c], channel_axis=None
            )
        # Blend to avoid over-transfer
        out_lab = (1 - strength) * content_lab + strength * transferred
        return lab_to_rgb(out_lab)

    # With mask → only match pixels inside mask
    out_lab = content_lab.copy()
    region = mask.astype(bool)

    for c in range(3):
        content_region = content_lab[..., c]
        style_region = style_lab[..., c]

        matched = match_histograms(
            content_region[region], style_region[region], channel_axis=None
        )

        # insert back
        temp = content_region.copy()
        temp[region] = matched

        # blend
        out_lab[..., c] = (1 - strength) * content_region + strength * temp

    return lab_to_rgb(out_lab)


def apply_per_region_transfer(content, style, seg_mask, region_ids, alpha=1.0):
    """
    Per-region histogram-based color transfer.
    """
    blended = content.copy()

    for region_id in region_ids:
        region_mask = seg_mask == region_id

        transferred = match_color_stats(
            content,
            style,
            mask=region_mask,
            strength=1.0    # strong color transfer
        )

        # Local blending
        blended[region_mask] = (
            alpha * transferred[region_mask] +
            (1 - alpha) * blended[region_mask]
        )

    return blended

# Example usage
if __name__ == "__main__":
    content_path = "../../Data/content/house.jpg"
    style_path = "../../Data/style/starry_night.jpg"
    output_path = "../../Data/results/color_transfer_sample1.jpg"

    # Load images
    content = load_image(content_path)
    style = load_image(style_path)

    # Perform global color transfer
    transferred = match_color_stats(content, style)
    save_image(output_path, transferred)
    print(f"Global color transfer completed and saved to {output_path}")

