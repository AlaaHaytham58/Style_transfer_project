"""
TODO: Follow the algorithm pseudocode from the provided paper (1609.03057v3).

This module is intended to be the high-level runner for the style-transfer
pipeline. Implement the pipeline to follow the paper's pseudocode roughly as:

1) Parse inputs: content image C, style image S, optional segmentation masks.
2) Preprocess: convert images to working color space (e.g., Lab) and build
	 multi-scale image pyramids (coarse -> fine).
3) For each scale (coarsest to finest):
	 - Initialize or upsample the nearest-neighbor field (NNF) from previous
		 scale (or random initialization at coarsest scale).
	 - Run the inner optimization: alternate between PatchMatch (NN search)
		 and reconstruction/voting/color-transfer steps as in the paper.
	 - Optionally run per-segmentation color transfer and blending.
4) Postprocess: convert back to display color space, save output.

Mapping to files in this project:
 - `patch_match.py` should implement the PatchMatch ANN search used in step 3.
 - `style_transfer.py` should implement the multi-scale orchestration and the
	 EM-like loop (match + reconstruction + color transfer).
 - `utils/color_transfer.py` performs color-statistics matching (Lab mean/std
	 or per-region matching) described in the paper.
 - `utils/io.py` and `utils/visualization.py` provide helpers for image I/O
	 and debugging visualizations.

Parameters to expose and tune (match paper names):
 - patch_size(s), num_scales, num_iterations_per_scale, propagation_iter,
	 random_search_radius, alpha (mixing), color_transfer_strength.

TODOs for implementation:
 - Implement a controlled multi-scale loop with deterministic upsample rules.
 - Ensure PatchMatch returns both NN indices and match distances.
 - Keep data in float32 normalized [0,1] for numerics; convert to uint8 only
	 when saving.
 - Add unit tests for one-scale behaviour (PatchMatch + reconstruction).
"""

"""
Simple CLI entry point to run the style transfer implemented in src/style_transfer.py

Usage (from project root):
> python main.py --content path\to\content.jpg --style path\to\style.jpg --out out.png
"""
import os
import sys
import time
import argparse
from skimage import io
import numpy as np

# ensure src/ is on path so we can import style_transfer
ROOT = os.path.dirname(os.path.dirname(__file__))  # project root (parent of src/)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from style_transfer import style_transfer  # type: ignore


def _load_image(path):
    img = io.imread(path)
    # drop alpha if present
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]
    return img


def _save_image(path, img):
    img = np.asarray(img)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    io.imsave(path, img)


def parse_args():
    p = argparse.ArgumentParser(description="Run patch-based multi-scale style transfer")
    p.add_argument("--scales", type=int, default=5, help="Number of pyramid scales")
    p.add_argument("--patch", type=int, default=7, help="Patch size (odd recommended)")
    p.add_argument("--iters", type=int, default=5, help="PatchMatch iterations per scale")
    p.add_argument("--no-color", action="store_true", help="Disable final color transfer (useful for debugging texture)")
    return p.parse_args()


def main():
    args = parse_args()
    content = _load_image("./Data/content/house.jpg")
    style = _load_image("./Data/style/starry_night.jpg")
    print(f"Content: {content.shape}, Style: {style.shape}")
    t0 = time.time()
    out = style_transfer(content, style, num_scales=args.scales,
                         patch_size=args.patch, pm_iters=args.iters, 
                         apply_color_transfer=(not args.no_color), debug=True,
                         style_max_dim=100, tile_style=True)
    duration = time.time() - t0
    if out is None:
        print("Style transfer returned None -- aborting.")
        return
    _save_image("res.jpg", out)
    print(f"Saved result to res.jpg (took {duration:.1f}s)")


if __name__ == "__main__":
    main()