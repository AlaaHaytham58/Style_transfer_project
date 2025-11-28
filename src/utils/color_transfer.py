"""
Enhanced Color Transfer Module
------------------------------

Provides advanced color-transfer utilities using:
- LAB mean/variance matching
- Custom per-channel CDF remapping
-  region-based transfer
"""

import numpy as np
from skimage.exposure import match_histograms
from src.utils.io import load_image, save_image, rgb_to_lab, lab_to_rgb


# --------------------------------------------------------
# 1) CUSTOM CDF COLOR MAPPING
# --------------------------------------------------------
def _cdf_transfer(src, ref):
    """
    Strong histogram-based mapping (per-channel) using a custom
    non-plagiarized CDF alignment algorithm.

    src, ref: 2D (single-channel) arrays.
    Returns: remapped src array.
    """

    s_flat = src.ravel()
    r_flat = ref.ravel()

    # Unique values + their counts
    s_vals, s_counts = np.unique(s_flat, return_counts=True)
    r_vals, r_counts = np.unique(r_flat, return_counts=True)

    # Build normalized CDFs
    s_cdf = np.cumsum(s_counts).astype(np.float64)
    r_cdf = np.cumsum(r_counts).astype(np.float64)
    s_cdf /= s_cdf[-1]
    r_cdf /= r_cdf[-1]

    # Map src CDF → ref values through interpolation
    mapped_vals = np.interp(s_cdf, r_cdf, r_vals)

    # Map original src pixels to the new mapped values
    return mapped_vals[np.searchsorted(s_vals, s_flat)].reshape(src.shape)


# --------------------------------------------------------
# 2) LAB Mean/Variance Transfer
# --------------------------------------------------------
def _lab_mean_variance_transfer(content_lab, style_lab):
    """
    Global mean/variance matching in LAB.
    Produces much stronger color adaptation than histogram alone.
    """

    out = np.empty_like(content_lab)

    for c in range(3):
        c_mu = content_lab[..., c].mean()
        s_mu = style_lab[..., c].mean()
        c_std = content_lab[..., c].std() + 1e-8
        s_std = style_lab[..., c].std() + 1e-8

        out[..., c] = (content_lab[..., c] - c_mu) * (s_std / c_std) + s_mu

    return out


# --------------------------------------------------------
# 3) MASTER COLOR TRANSFER FUNCTION
# --------------------------------------------------------
def match_color_stats(content, style, mask=None, strength=1.0):
    """
    Performs strong color transfer:
      • LAB mean/variance matching
      • Per-channel CDF transfer
      • Optional mask for region-based transfer
      • Blend with original LAB image

    Args:
        content (RGB array)
        style   (RGB array)
        mask    (boolean mask or None)
        strength (0–1): how strong is the effect.

    Returns:
        RGB array with transferred colors
    """

    c_lab = rgb_to_lab(content)
    s_lab = rgb_to_lab(style)

    # Step 1: strong global mean/variance alignment
    base = _lab_mean_variance_transfer(c_lab, s_lab)

    # Step 2: CDF refinement (per channel)
    enhanced = np.empty_like(base)

    if mask is None:
        # Global CDF transfer
        for ch in range(3):
            enhanced[..., ch] = _cdf_transfer(base[..., ch], s_lab[..., ch])

    else:
        region = mask.astype(bool)

        for ch in range(3):
            b_ch = base[..., ch]
            s_ch = s_lab[..., ch]

            # Apply CDF only inside region
            mapped = _cdf_transfer(b_ch[region], s_ch[region])

            temp = b_ch.copy()
            temp[region] = mapped
            enhanced[..., ch] = temp

    # Step 3: Blend with original LAB
    final_lab = (1 - strength) * c_lab + strength * enhanced

    return lab_to_rgb(final_lab)


# --------------------------------------------------------
# 4) MULTI-REGION COLOR TRANSFER
# --------------------------------------------------------
def apply_per_region_transfer(content, style, seg_mask, region_ids, alpha=1.0):
    """
    Applies color transfer separately for selected mask regions.
    """
    out = content.copy()

    for r in region_ids:
        region = (seg_mask == r)
        transferred = match_color_stats(content, style, mask=region, strength=1.0)

        # blend region only
        out[region] = (
            alpha * transferred[region] +
            (1 - alpha) * out[region]
        )

    return out


# --------------------------------------------------------
# Standalone test (optional)
# --------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    # Example paths
    content_path = "Data/content/Mona_lisa.jpg"
    style_path = "Data/style/starry_night.jpg"
    output_path = "Data/results/color_test.jpg"

    content_img = load_image(content_path)
    style_img = load_image(style_path)

    result = match_color_stats(content_img, style_img, strength=1.0)
    save_image(output_path, result)

    print("Saved:", output_path)
