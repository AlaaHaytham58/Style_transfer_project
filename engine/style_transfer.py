import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


class StyleTransfer:

    def __init__(
        self,
        patch_sizes,
        gaps,
        r_robust,
        irls_iterations,
        em_iterations_per_patch,
        num_levels
    ):
        self.patch_sizes = patch_sizes
        self.gaps = gaps
        self.r = r_robust
        self.irls_iterations = irls_iterations
        self.em_iterations = em_iterations_per_patch
        self.num_levels = num_levels

    def extract_patches_with_gap(self, image, patch_size, gap):
        """
        Extract patches with specified gap (subsampling).

        Args:
            image: Input image
            patch_size: Size of square patches
            gap: Stride between patches (subsampling)

        Returns:
            patches: Flattened patches
            positions: (i, j) positions of top-left corners
        """
        h, w = image.shape[:2]
        patches = []
        positions = []

        for i in range(0, h - patch_size + 1, gap):
            for j in range(0, w - patch_size + 1, gap):
                patch = image[i:i + patch_size, j:j + patch_size]
                patches.append(patch.flatten())
                positions.append((i, j))

        return np.array(patches), positions

    def find_nearest_neighbors(self, content_patches, style_patches):
        """Find nearest neighbor in style for each content patch"""
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(style_patches)
        _, indices = nbrs.kneighbors(content_patches)
        return indices.flatten()

    def robust_aggregate_IRLS(
        self,
        matched_patches,
        positions,
        output_shape,
        patch_size,
        content=None,
        W=None
    ):
        """
        IRLS patch aggregation with robust norm r.

        Implements Equations (8–11) from the paper.
        """
        h, w, c = output_shape

        X = content.copy() if content is not None else \
            np.random.rand(h, w, c).astype(np.float32) * 0.1

        for _ in range(self.irls_iterations):
            weights = []

            for patch_flat, (i, j) in zip(matched_patches, positions):
                patch = patch_flat.reshape(patch_size, patch_size, c)

                h_end = min(i + patch_size, h)
                w_end = min(j + patch_size, w)

                current_region = X[i:h_end, j:w_end]
                error = np.linalg.norm(
                    current_region - patch[:h_end - i, :w_end - j]
                )

                weight = np.power(error + 1e-8, self.r - 2)
                weights.append(weight)

            weights = np.array(weights)

            numerator = np.zeros((h, w, c), dtype=np.float32)
            denominator = np.zeros((h, w, 1), dtype=np.float32)

            for weight, patch_flat, (i, j) in zip(weights, matched_patches, positions):
                patch = patch_flat.reshape(patch_size, patch_size, c)

                h_end = min(i + patch_size, h)
                w_end = min(j + patch_size, w)

                numerator[i:h_end, j:w_end] += weight * patch[:h_end - i, :w_end - j]
                denominator[i:h_end, j:w_end] += weight

            X_tilde = numerator / np.maximum(denominator, 1e-8)

            if content is not None and W is not None:
                if W.ndim == 2:
                    W = W[:, :, np.newaxis]
                if W.shape[:2] != (h, w):
                    W = cv2.resize(W, (w, h))
                    W = W[:, :, np.newaxis]

                X = (X_tilde + W * content) / (1 + W)
            else:
                X = X_tilde

            X = np.clip(X, 0, 1)

        return X

    def create_edge_segmentation(self, content):
        """Create simple edge-based segmentation mask"""
        gray = cv2.cvtColor(
            (content * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)
        mask = gaussian_filter(mask.astype(np.float32) / 255.0, sigma=2.0)
        return mask

    def process_with_patch_size(
        self,
        content_level,
        style_level,
        patch_size,
        gap,
        prev_result=None,
        W=None
    ):
        """
        Process one pyramid level with one patch size.
        """
        h, w = content_level.shape[:2]

        if prev_result is None:
            X = content_level + np.random.randn(*content_level.shape) * (50.0 / 255.0)
            X = np.clip(X, 0, 1).astype(np.float32)
        else:
            X = cv2.resize(prev_result, (w, h)) \
                if prev_result.shape[:2] != (h, w) else prev_result.copy()

        style_patches, _ = self.extract_patches_with_gap(style_level, patch_size, gap)

        for _ in range(self.em_iterations):
            X_patches, positions = self.extract_patches_with_gap(X, patch_size, gap)
            nn_indices = self.find_nearest_neighbors(X_patches, style_patches)
            matched_patches = style_patches[nn_indices]

            X = self.robust_aggregate_IRLS(
                matched_patches,
                positions,
                (h, w, 3),
                patch_size,
                content_level,
                W
            )

            X = gaussian_filter(X, sigma=0.5)
            X = np.clip(X, 0, 1)

        return X
    def color_transfer(
        self,
        content,
        style,
        mask=None,
        strength=0.7,
        chroma_boost=1.1,
        style_chroma_boost=1.05,
        cdf_passes=1,
        gamma=1.0,
        enhance_L_contrast=True,
    ):
        """
        Unified enhanced color transfer:
        - LAB mean/variance alignment
        - Multi-pass CDF refinement
        - Optional region masking
        - Contrast & gamma enhancement

        content, style: RGB images in [0,1]
        returns: RGB image in [0,1]
        """

        # Local helper: CDF mapping
        def cdf_transfer(src, ref):
            s_flat = src.ravel()
            r_flat = ref.ravel()

            s_vals, s_counts = np.unique(s_flat, return_counts=True)
            r_vals, r_counts = np.unique(r_flat, return_counts=True)

            s_cdf = np.cumsum(s_counts).astype(np.float64)
            r_cdf = np.cumsum(r_counts).astype(np.float64)
            s_cdf /= s_cdf[-1]
            r_cdf /= r_cdf[-1]

            mapped_vals = np.interp(s_cdf, r_cdf, r_vals)
            return mapped_vals[np.searchsorted(s_vals, s_flat)].reshape(src.shape)

 
        # Local helper: LAB mean/variance step
 
        def lab_mean_variance_transfer(c_lab, s_lab):
            out = np.empty_like(c_lab)

            for ch in range(3):
                c_chan = c_lab[..., ch]
                s_chan = s_lab[..., ch]

                out[..., ch] = (
                    (c_chan - c_chan.mean())
                    * (s_chan.std() / (c_chan.std() + 1e-8))
                    + s_chan.mean()
                )

            # Boost chroma & luminance
            out[..., 0] *= 1.0           # L
            out[..., 1] *= chroma_boost  # a
            out[..., 2] *= chroma_boost  # b

            out[..., 0] = np.clip(out[..., 0], 0, 100)
            out[..., 1:] = np.clip(out[..., 1:], -128, 127)

            return out

         
        # 1. RGB → LAB
         
        c_lab = cv2.cvtColor(content, cv2.COLOR_RGB2LAB).astype(np.float32)
        s_lab = cv2.cvtColor(style, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Exaggerate style chroma
        s_lab[..., 1:] *= style_chroma_boost
        s_lab[..., 1:] = np.clip(s_lab[..., 1:], -128, 127)

         
        # 2. Mean / Variance match
         
        enhanced = lab_mean_variance_transfer(c_lab, s_lab)

         
        # 3. Multi-pass CDF refine
         
        for _ in range(max(1, cdf_passes)):
            new_lab = enhanced.copy()

            for ch in range(3):
                if mask is None:
                    new_lab[..., ch] = cdf_transfer(
                        enhanced[..., ch], s_lab[..., ch]
                    )
                else:
                    region = mask.astype(bool)
                    tmp = enhanced[..., ch]
                    tmp[region] = cdf_transfer(
                        tmp[region], s_lab[..., ch][region]
                    )
                    new_lab[..., ch] = tmp

            enhanced = new_lab

         
        # 4. Blend with original
         
        final_lab = (1.0 - strength) * c_lab + strength * enhanced

        
        # 5. L-contrast stretch
        if enhance_L_contrast:
            L = final_lab[..., 0]
            L_min, L_max = L.min(), L.max()
            if L_max > L_min + 1e-6:
                final_lab[..., 0] = (L - L_min) / (L_max - L_min) * 100.0

        # 6. LAB → RGB + gamma
        result = cv2.cvtColor(final_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        result = np.clip(result, 0.0, 1.0)

        if gamma != 1.0:
            result = np.power(result, gamma)
            result = np.clip(result, 0.0, 1.0)

        return result


    def build_pyramid(self, image, num_levels):
        """Build Gaussian pyramid"""
        pyramid = [image]
        for _ in range(num_levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid[::-1]  # Coarse to fine

