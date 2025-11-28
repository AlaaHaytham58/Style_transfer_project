"""
Enhanced Color Transfer Module
------------------------------

Provides advanced color-transfer utilities using:
- LAB mean/variance matching (with chroma boosting)
- Custom per-channel CDF remapping (multi-pass)
- Optional region-based transfer
- Contrast & gamma enhancement

Author: You :)
"""

import numpy as np
from src.utils.io import load_image, save_image, rgb_to_lab, lab_to_rgb


# --------------------------------------------------------
# 1) CUSTOM CDF COLOR MAPPING
# --------------------------------------------------------
def _cdf_transfer(src, ref):
    """
    Strong histogram-based mapping (per-channel) using a custom
    CDF alignment algorithm.

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
# 2) LAB Mean/Variance Transfer (with chroma boost)
# --------------------------------------------------------
def _lab_mean_variance_transfer(content_lab, style_lab, chroma_boost=1.5, l_boost=1.0):
    """
    Global mean/variance matching in LAB + chroma boosting.

    chroma_boost > 1.0 → stronger a/b (color) channels
    l_boost > 1.0      → stronger luminance range
    """

    out = np.empty_like(content_lab)

    for c in range(3):
        c_chan = content_lab[..., c]
        s_chan = style_lab[..., c]

        c_mu = c_chan.mean()
        s_mu = s_chan.mean()
        c_std = c_chan.std() + 1e-8
        s_std = s_chan.std() + 1e-8

        out[..., c] = (c_chan - c_mu) * (s_std / c_std) + s_mu

    # Apply boosts
    out[..., 0] *= l_boost                # L channel
    out[..., 1] *= chroma_boost           # a channel
    out[..., 2] *= chroma_boost           # b channel

    # Clip to a reasonable LAB range
    out[..., 0] = np.clip(out[..., 0], 0, 100)        # L in [0,100]
    out[..., 1] = np.clip(out[..., 1], -128, 127)     # a in [-128,127]
    out[..., 2] = np.clip(out[..., 2], -128, 127)     # b in [-128,127]

    return out


# --------------------------------------------------------
# 3) MASTER COLOR TRANSFER FUNCTION (super strong)
# --------------------------------------------------------
def match_color_stats(
    content,
    style,
    mask=None,
    strength=1.0,
    chroma_boost=1.5,
    style_chroma_boost=1.2,
    cdf_passes=2,
    gamma=0.9,
    enhance_L_contrast=True,
):
    """
    Performs strong color transfer:
      • LAB mean/variance matching (with chroma_boost)
      • Multi-pass per-channel CDF transfer
      • Optional mask for region-based transfer
      • L-contrast stretching
      • Gamma correction in RGB
      • Blend with original LAB image

    Args:
        content  (RGB array)
        style    (RGB array)
        mask     (boolean mask or None)
        strength (0–1): how strong to lean towards style.
        chroma_boost: >1 → more saturated result.
        style_chroma_boost: >1 → exaggerate style colors.
        cdf_passes: 1 or 2 for stronger histogram alignment.
        gamma: <1 → brighter & punchier midtones.
        enhance_L_contrast (bool): whether to stretch L.

    Returns:
        RGB array with transferred colors
    """

    # Convert to LAB
    c_lab = rgb_to_lab(content)
    s_lab = rgb_to_lab(style)

    # Exaggerate style chroma a bit to make palette more aggressive
    s_lab[..., 1] *= style_chroma_boost
    s_lab[..., 2] *= style_chroma_boost
    s_lab[..., 1] = np.clip(s_lab[..., 1], -128, 127)
    s_lab[..., 2] = np.clip(s_lab[..., 2], -128, 127)

    # Step 1: strong global mean/variance alignment with chroma boost
    base = _lab_mean_variance_transfer(c_lab, s_lab, chroma_boost=chroma_boost, l_boost=1.0)

    # Step 2: multi-pass CDF refinement (per channel)
    enhanced = base.copy()

    for _ in range(max(1, cdf_passes)):
        new_enhanced = np.empty_like(enhanced)

        if mask is None:
            # Global CDF transfer
            for ch in range(3):
                new_enhanced[..., ch] = _cdf_transfer(enhanced[..., ch], s_lab[..., ch])
        else:
            region = mask.astype(bool)

            for ch in range(3):
                b_ch = enhanced[..., ch]
                s_ch = s_lab[..., ch]

                mapped = _cdf_transfer(b_ch[region], s_ch[region])

                temp = b_ch.copy()
                temp[region] = mapped
                new_enhanced[..., ch] = temp

        enhanced = new_enhanced

    # Step 3: blend with original LAB
    # You can add an extra factor here if you want even more aggression.
    final_lab = (1.0 - strength) * c_lab + strength * enhanced

    # Optional: enhance L contrast by stretching it
    if enhance_L_contrast:
        L = final_lab[..., 0]
        L_min = L.min()
        L_max = L.max()
        if L_max > L_min + 1e-6:
            L = (L - L_min) / (L_max - L_min) * 100.0
            final_lab[..., 0] = L

    # Convert back to RGB
    result = lab_to_rgb(final_lab)

    # Gamma correction for extra punch
    if gamma != 1.0:
        result = np.clip(result, 0.0, 1.0)
        result = np.power(result, gamma)
        result = np.clip(result, 0.0, 1.0)

    return result


# --------------------------------------------------------
# 4) MULTI-REGION COLOR TRANSFER
# --------------------------------------------------------
def apply_per_region_transfer(content, style, seg_mask, region_ids, alpha=1.0, **kwargs):
    """
    Applies color transfer separately for selected mask regions.

    kwargs are forwarded to match_color_stats (e.g. chroma_boost, cdf_passes, etc.)
    """
    out = content.copy()

    for r in region_ids:
        region = (seg_mask == r)
        transferred = match_color_stats(content, style, mask=region, strength=1.0, **kwargs)

        # blend region only
        out[region] = alpha * transferred[region] + (1.0 - alpha) * out[region]

    return out


# --------------------------------------------------------
# Standalone test
# --------------------------------------------------------
if __name__ == "__main__":
    # IMPORTANT: run from project root:
    #   python -m src.utils.color_transfer
    content_path = "Data/content/Mountain.jpg"
    style_path = "Data/style/styled_candy.jpg"
    output_path = "Data/results/color_test_new.jpg"

    content_img = load_image(content_path)
    style_img = load_image(style_path)

    result = match_color_stats(
        content_img,
        style_img,
        strength=1.0,          # full style influence
        chroma_boost=1.8,      # more saturation on output
        style_chroma_boost=2.8,
        cdf_passes=3,          # very strong histogram forcing
        gamma=0.9,             # punchier midtones
        enhance_L_contrast=True,
    )

    save_image(output_path, result)
    print("Saved:", output_path)
