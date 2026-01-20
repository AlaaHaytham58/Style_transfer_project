"""
TODO: Implement the EM-like style-transfer driver following the paper's
pseudocode.

Suggested responsibilities for this file:
 - Build image pyramids for content and style (multi-scale with downsample
	and upsample rules).
 - For each scale, perform the inner optimization loop:
	  a) Initialize/upsample the NNF.
	  b) Run a fixed number of PatchMatch iterations (call into
		  `patch_match.find_ann` or similar).
	  c) Reconstruct the output image from the NNF (voting or averaging
		  overlapping patches).
	  d) Optionally apply color transfer (call `utils.color_transfer`), and
		  enforce segmentation constraints if masks available.
 - Manage weights and schedule (how and when to apply color_transfer,
	strength alpha, etc.).

Key functions to provide (recommended):
 - build_pyramid(image, num_scales)
 - run_scale(content_scale, style_scale, init_nnf=None)
 - reconstruct_from_nnf(nnf, style_image, patch_size)
 - upsample_nnf(nnf, out_shape)

Edge cases & checks:
 - Ensure patch extraction respects image boundaries (use padding or valid
	masks).
 - Validate that NNF coordinates are clamped inside source image.
"""
# ...existing code...
"""
Multi-scale patch-based style transfer driver.

Implements:
 - build_pyramid(image, num_scales)
 - run_scale(content_scale, style_scale, init_nnf=None)
 - reconstruct_from_nnf(nnf, style_image, patch_size)
 - upsample_nnf(nnf, out_shape)

Fallbacks:
 - Simple random-search-based NNF improvement if a project-local patch_match
   implementation is not available.
 - Optional color transfer via utils.color_transfer if utils is available.

All images are treated as floats in [0,1] internally.
"""
from typing import List, Tuple, Optional
import numpy as np
from skimage.transform import resize
    
import patch_match
import utils  # may provide color_transfer
 

def _to_float01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def build_pyramid(image: np.ndarray, num_scales: int, min_size: int = 16) -> List[np.ndarray]:
    """
    Build an image pyramid from coarse -> fine.
    """
    img = _to_float01(image)
    pyramid = [img]
    while len(pyramid) < num_scales:
        h, w = pyramid[-1].shape[:2]
        if h // 2 < min_size or w // 2 < min_size:
            break
        nxt = resize(pyramid[-1], (h // 2, w // 2), order=1, preserve_range=True, anti_aliasing=True)
        pyramid.append(nxt.astype(np.float32))
    return list(reversed(pyramid))


def _clamp_coords(y: int, x: int, H: int, W: int, patch_size: int) -> Tuple[int, int]:
    return int(np.clip(y, 0, max(0, H - patch_size))), int(np.clip(x, 0, max(0, W - patch_size)))


def _init_random_nnf(content_shape: Tuple[int, int], style_shape: Tuple[int, int], patch_size: int) -> np.ndarray:
    Hc, Wc = content_shape
    Hy = max(1, Hc - patch_size + 1)
    Wx = max(1, Wc - patch_size + 1)
    Hs, Ws = style_shape
    max_y = max(1, Hs - patch_size + 1)
    max_x = max(1, Ws - patch_size + 1)
    rand_y = np.random.randint(0, max_y, size=(Hy, Wx))
    rand_x = np.random.randint(0, max_x, size=(Hy, Wx))
    nnf = np.stack([rand_y, rand_x], axis=2).astype(np.int32)
    return nnf


def upsample_nnf(nnf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upsample NNF using nearest-neighbor mapping (no coordinate scaling).
    Keeps the same style coordinates from the nearest coarse cell.
    """
    ny_out, nx_out = out_shape
    ny_in, nx_in = nnf.shape[:2]
    if ny_in == ny_out and nx_in == nx_out:
        return nnf.copy()
    scale_y = ny_in / float(ny_out)
    scale_x = nx_in / float(nx_out)
    yy = (np.arange(ny_out) * scale_y).astype(int)
    xx = (np.arange(nx_out) * scale_x).astype(int)
    yy = np.clip(yy, 0, ny_in - 1)
    xx = np.clip(xx, 0, nx_in - 1)
    up = np.zeros((ny_out, nx_out, 2), dtype=nnf.dtype)
    for i, yi in enumerate(yy):
        for j, xj in enumerate(xx):
            y_src, x_src = nnf[yi, xj]
            # Do NOT multiply coordinates; keep original style coords
            up[i, j, 0] = int(y_src)
            up[i, j, 1] = int(x_src)
    return up


def reconstruct_from_nnf(nnf: np.ndarray, style_image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Reconstruct an image by averaging overlapping patches from style_image guided by nnf.
    Output size = (ny + patch_size -1, nx + patch_size -1)
    """
    Hs, Ws = style_image.shape[:2]
    ny, nx = nnf.shape[:2]
    out_h = ny + patch_size - 1
    out_w = nx + patch_size - 1
    C = style_image.shape[2] if style_image.ndim == 3 else 1
    out = np.zeros((out_h, out_w, C), dtype=np.float32)
    counts = np.zeros((out_h, out_w, 1), dtype=np.float32)

    pad = ((0, max(0, patch_size)), (0, max(0, patch_size)), (0, 0)) if C > 1 else ((0, max(0, patch_size)), (0, max(0, patch_size)))
    style_padded = np.pad(style_image, pad, mode='reflect')
    for i in range(ny):
        for j in range(nx):
            sy, sx = int(nnf[i, j, 0]), int(nnf[i, j, 1])
            sy, sx = _clamp_coords(sy, sx, Hs, Ws, patch_size)
            patch = style_padded[sy:sy + patch_size, sx:sx + patch_size]
            if C == 1 and patch.ndim == 2:
                patch = patch[:, :, None]
            out[i:i + patch_size, j:j + patch_size] += patch.astype(np.float32)
            counts[i:i + patch_size, j:j + patch_size] += 1.0
    counts[counts == 0] = 1.0
    out = out / counts
    out = np.clip(out, 0.0, 1.0)
    if C == 1:
        out = out[:, :, 0]
    return out


def _patch_ssd(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    return float(np.sum((patch_a.astype(np.float32) - patch_b.astype(np.float32)) ** 2))


def _random_search_nnf(content: np.ndarray, style: np.ndarray, nnf: np.ndarray, patch_size: int, num_candidates: int = 10, iters: int = 3) -> np.ndarray:
    Hc, Wc = content.shape[:2]
    Hy = max(1, Hc - patch_size + 1)
    Wx = max(1, Wc - patch_size + 1)
    Hs, Ws = style.shape[:2]
    style_padded = np.pad(style, ((0, patch_size), (0, patch_size), (0, 0)), mode='reflect')

    for _ in range(iters):
        for i in range(Hy):
            for j in range(Wx):
                cpatch = content[i:i + patch_size, j:j + patch_size]
                best_y, best_x = int(nnf[i, j, 0]), int(nnf[i, j, 1])
                best_y, best_x = _clamp_coords(best_y, best_x, Hs, Ws, patch_size)
                best_patch = style_padded[best_y:best_y + patch_size, best_x:best_x + patch_size]
                best_score = _patch_ssd(cpatch, best_patch)
                for _c in range(num_candidates):
                    ry = np.random.randint(0, max(1, Hs - patch_size + 1))
                    rx = np.random.randint(0, max(1, Ws - patch_size + 1))
                    cand = style_padded[ry:ry + patch_size, rx:rx + patch_size]
                    score = _patch_ssd(cpatch, cand)
                    if score < best_score:
                        best_score = score
                        best_y, best_x = ry, rx
                nnf[i, j, 0] = best_y
                nnf[i, j, 1] = best_x
    return nnf


def run_scale(content_scale: np.ndarray, style_scale: np.ndarray, init_nnf: Optional[np.ndarray] = None, patch_size: int = 7, num_pm_iters: int = 5, apply_color_transfer: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single-scale EM-like loop: initialize/upsample NNF, run PatchMatch (or fallback),
    reconstruct output, optionally color transfer.
    Returns (output_image, nnf)
    """
    content = _to_float01(content_scale)
    style = _to_float01(style_scale)
    Hc, Wc = content.shape[:2]
    Hy = max(1, Hc - patch_size + 1)
    Wx = max(1, Wc - patch_size + 1)

    if init_nnf is None:
        nnf = _init_random_nnf((Hc, Wc), style.shape[:2], patch_size)
    else:
        nnf = init_nnf.copy()
        nnf[..., 0] = np.clip(nnf[..., 0], 0, max(0, style.shape[0] - patch_size))
        nnf[..., 1] = np.clip(nnf[..., 1], 0, max(0, style.shape[1] - patch_size))

    if patch_match is not None and hasattr(patch_match, "find_ann"):
        try:
            nnf = patch_match.find_ann(content, style, patch_size=patch_size, init_ann=nnf, num_iter=num_pm_iters)
        except Exception:
            nnf = _random_search_nnf(content, style, nnf, patch_size, iters=num_pm_iters)
    else:
        nnf = _random_search_nnf(content, style, nnf, patch_size, iters=num_pm_iters)

    out = reconstruct_from_nnf(nnf, style, patch_size)

    if apply_color_transfer and utils is not None and hasattr(utils, "color_transfer"):
        try:
            out = utils.color_transfer(content, out)
            out = np.clip(out, 0.0, 1.0)
        except Exception:
            pass

    return out, nnf


def style_transfer(content: np.ndarray, style: np.ndarray, num_scales: int = 5, patch_size: int = 7, pm_iters: int = 5) -> np.ndarray:
    """
    Full multi-scale driver. Color transfer applied only on the final (finest) scale.
    """
    content = _to_float01(content)
    style = _to_float01(style)
    cpyr = build_pyramid(content, num_scales)
    spyr = build_pyramid(style, num_scales)
    nnf = None
    out = None
    for level in range(len(cpyr)):
        c = cpyr[level]
        s = spyr[level]
        Hc, Wc = c.shape[:2]
        ny = max(1, Hc - patch_size + 1)
        nx = max(1, Wc - patch_size + 1)
        if nnf is not None:
            nnf = upsample_nnf(nnf, (ny, nx))
        out, nnf = run_scale(c, s, init_nnf=nnf, patch_size=patch_size, num_pm_iters=pm_iters, apply_color_transfer=(level == len(cpyr) - 1))
    return out

