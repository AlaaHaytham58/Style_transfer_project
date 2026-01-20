# engine/metrics.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _match_size(src, target):
    """
    Resize src image to match target spatial dimensions.
    """
    return cv2.resize(src, (target.shape[1], target.shape[0]))


def compute_metrics(content, style, output):
    """
    Compute torch-free quantitative metrics for style transfer evaluation.

    All images are uint8 RGB.
    Images are internally resized to match output resolution.
    """

    # ======================
    # Align spatial sizes
    # ======================
    if content.shape != output.shape:
        content = _match_size(content, output)

    if style.shape != output.shape:
        style = _match_size(style, output)

    # ======================
    # SSIM (content preservation)
    # ======================
    gray_c = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
    gray_o = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

    ssim_score = ssim(gray_c, gray_o, data_range=255)

    # ======================
    # Color histogram similarity (style similarity)
    # ======================
    hist_style = cv2.calcHist(
        [style],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    hist_out = cv2.calcHist(
        [output],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    cv2.normalize(hist_style, hist_style)
    cv2.normalize(hist_out, hist_out)

    hist_score = cv2.compareHist(
        hist_style,
        hist_out,
        cv2.HISTCMP_CORREL
    )

    return {
        "SSIM (Content)": round(float(ssim_score), 4),
        "Color Corr (Style)": round(float(hist_score), 4)
    }
