# engine/pipeline.py
import cv2
import numpy as np
from .style_transfer import StyleTransfer
from engine.metrics import compute_metrics


def _resize_max(img, max_size):
    """
    Resize image while preserving aspect ratio.
    """
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def run_style_transfer(
    content_img,
    style_img,
    *,
    patch_sizes,
    gaps,
    r_robust,
    irls_iterations,
    em_iterations,
    num_levels,
    max_size,
    use_segmentation=True,
    final_color_transfer=True
):
    """
    Runs patch-based style transfer and returns:
      - output image (uint8)
      - evaluation metrics (dict)
    """

    # ======================
    # 1. Normalize
    # ======================
    content = content_img.astype(np.float32) / 255.0
    style = style_img.astype(np.float32) / 255.0

    # ======================
    # 2. Resize (aspect-ratio preserved)
    # ======================
    content = _resize_max(content, max_size)
    style = _resize_max(style, max_size)

    # ======================
    # 3. Initialize engine
    # ======================
    if len(patch_sizes) != len(gaps):
        raise ValueError("patch_sizes and gaps must have the same length")

    engine = StyleTransfer(
        patch_sizes,
        gaps,
        r_robust,
        irls_iterations,
        em_iterations,
        num_levels
    )

    # ======================
    # 4. Segmentation
    # ======================
    W = engine.create_edge_segmentation(content) if use_segmentation else None

    # ======================
    # 5. Initial color transfer
    # ======================
    content = engine.color_transfer(
        content,
        style,
        strength=0.9,
        chroma_boost=1.6,
        cdf_passes=2
    )

    # ======================
    # 6. Pyramids
    # ======================
    content_pyr = engine.build_pyramid(content, num_levels)
    style_pyr = engine.build_pyramid(style, num_levels)

    result = None

    # ======================
    # 7. Patch optimization
    # ======================
    for level in range(num_levels):
        c = content_pyr[level]
        s = style_pyr[level]

        if W is not None:
            W_level = cv2.resize(
                W,
                (c.shape[1], c.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            W_level = None

        for p, g in zip(patch_sizes, gaps):
            if p < min(c.shape[:2]):
                result = engine.process_with_patch_size(
                    c,
                    s,
                    p,
                    g,
                    prev_result=result,
                    W=W_level
                )

    if result is None:
        raise RuntimeError(
            "Style transfer failed: no valid patch size applied."
        )

    # ======================
    # 8. Final color transfer (optional)
    # ======================
    if final_color_transfer:
        result = engine.color_transfer(result, style)

    # ======================
    # 9. Convert to uint8
    # ======================
    output = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    # ======================
    # 10. Resize originals for evaluation
    # ======================
    content_eval = cv2.resize(
        content_img.astype(np.uint8),
        (output.shape[1], output.shape[0])
    )
    style_eval = cv2.resize(
        style_img.astype(np.uint8),
        (output.shape[1], output.shape[0])
    )

    # ======================
    # 11. Metrics
    # ======================
    metrics = compute_metrics(content_eval, style_eval, output)

    return output, metrics
