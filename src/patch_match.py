"""
TODO: Implement PatchMatch nearest-neighbor search per the paper pseudocode.

Core algorithm steps (map these to functions and tests):
 1) Initialization:
		- For each target patch location, initialize a random nearest-neighbor
			(x,y) in the source/style image or upsample from coarse-level NNF.
		- Also store the patch distance (e.g., SSD in a chosen feature space).

 2) Iterative improvement (repeat for several iterations):
		For iteration t in 0..T-1:
			- Choose scan order: if t is even, scan top-left -> bottom-right; if
				odd, scan bottom-right -> top-left (this is crucial for propagation).
			- Propagation: for each pixel p in scan order, try to improve p's NNF
				by testing neighbors' offsets (e.g., left/up or right/down depending
				on scan direction) and accept if distance improves.
			- Random search: for each pixel p, perform decreasing-radius random
				offsets around the current best match to escape local minima.

 3) Distance metric:
		- Use block-wise SSD in the chosen color or feature space (Lab, or CNN
			features if extended). Optionally apply gaussian weighting in patch.

 4) Multi-scale support:
		- When moving to finer scale, upsample the NNF (and scale coordinates)
			and recompute distances or re-evaluate.

Expected outputs and API:
 - find_ann(target, source, patch_size, num_iters, init_nnf=None) -> (nnf,
	 distances)
 - nnf shape: (H_target, W_target, 2) holding x,y coordinates into source.
 - distances shape: (H_target, W_target) per-patch matching cost.

TODOs for implementation:
 - Vectorize patch distance computation for speed where possible.
 - Add unit tests that verify propagation decreases distance monotonically
	 on a simple synthetic example.
 - Keep random seed option for reproducibility.
"""

