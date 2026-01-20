
import numpy as np
import cv2
from PIL import Image
from typing import Tuple
# IMAGE CONVERSION
def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 NumPy RGB image [0,1]."""
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0

def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert NumPy RGB image [0,1] or [0,255] to PIL Image."""
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# RESIZING
def resize_to_max_size(
    img: np.ndarray,
    max_size: int,
    interpolation=cv2.INTER_AREA
) -> np.ndarray:
    """
    Resize image so that max(h, w) == max_size while keeping aspect ratio.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)
def resize_exact(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image to (size, size)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
# VALIDATION
def validate_patch_config(patch_sizes, gaps):
    if len(patch_sizes) != len(gaps):
        raise ValueError("patch_sizes and gaps must have same length")

    for p, g in zip(patch_sizes, gaps):
        if g >= p:
            raise ValueError(f"Gap {g} must be smaller than patch size {p}")
def ensure_3_channels(img: np.ndarray) -> np.ndarray:
    """Ensure image has 3 channels."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img

def clip01(img: np.ndarray) -> np.ndarray:
    """Clip image to [0,1]."""
    return np.clip(img, 0.0, 1.0)

def clip255(img: np.ndarray) -> np.ndarray:
    """Clip image to [0,255] uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)
