import numpy as np
from sklearn.decomposition import PCA
from utils.io import load_image , build_pyramid
from skimage.color import rgb2lab, lab2rgb
#  rgb2lab: لتحويل الصور إلى فضاء اللون Lab  الذي يفصل سطوع الصورة عن اللون
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import matplotlib.pyplot as plt

#  The patch-sizes are [33, 21, 13, and 9].
#  The sub-sampling gaps are [28, 18, 8, and 5].
# def patchmatch_ann_subsampled(
#     input_img, style_img, patch_size=9, stride=5, num_iters=3, pca_energy=0.95,
#     feature_space='rgb',        
#     use_gaussian_weight=False,  
#     init_nnf=None,              
#     random_seed=None
# ):
#     """
#     Parameters:
#         input_img: (H, W, C) array – current estimate of output image X
#         style_img: (Hs, Ws, C) array – style image S
#         patch_size: int – e.g., 33, 21, 13, 9
#         stride: int – subsampling step in Ω grid
#         num_iters: int – number of PatchMatch iterations (Ialg = 3 per paper)
#         pca_energy: float – fraction of variance to preserve (default: 0.95)

#     Returns:
#         matches: dict (i, j) -> (k, l) in style image coordinates
#         distances: dict (i, j) -> SSD in original RGB space (for consistency)
#     """
#      # --- Handle feature_space conversion if needed ---
#     if feature_space == 'lab' and input_img.shape[2] == 3 and style_img.shape[2] == 3:
#         input_img_f = rgb2lab(input_img.astype(np.float32) / 255.0)
#         style_img_f = rgb2lab(style_img.astype(np.float32) / 255.0)
#     else:
#         input_img_f = input_img.astype(np.float32)
#         style_img_f = style_img.astype(np.float32)

#     # --- Random generator with optional seed ---
#     rng = np.random.default_rng(random_seed)
    
#     H, W = input_img.shape[:2]
#     Hs, Ws = style_img.shape[:2]
#     pad = patch_size // 2
#     C = input_img.shape[2] if input_img.ndim == 3 else 1

#     # Build list of all valid center coordinates in style image
#     style_centers = []
#     style_patches_flat = []
#     for l in range(pad, Hs - pad):
#         for k in range(pad, Ws - pad):
#             patch = style_img[l - pad:l + pad + 1, k - pad:k + pad + 1]
#             if patch.shape[:2] == (patch_size, patch_size):
#                 style_centers.append((k, l))
#                 style_patches_flat.append(patch.flatten())

#     if len(style_patches_flat) == 0:
#         raise ValueError("Style image too small for given patch_size")

#     #M عدد الـ patches من صورة الأسلوب.
# 	#بَدَل ما الـ patch تكون 2D نخليها 1D
#     P = np.stack(style_patches_flat, axis=1)  # shape: (n_pixels * C, M)
#     n_dim = P.shape[0]  # = patch_size * patch_size * C
#     M = P.shape[1]

#     # PCA: center and reduce
#     #keepdims=True يبقي الأبعاد كـ (n, 1) بدلًا من (n,) لتسهيل الطرح لاحقًا.
#     mP = np.mean(P, axis=1, keepdims=True)  # (n_dim, 1)
#     # تعريف التباين (variance) يفترض أن المتوسط = 0
#     P_centered = P - mP    #needed for PCA
    
# 	#     PCA = Principal Component Analysis. #
# 	# خوارزمية تُستخدم لتقليل الأبعاد مع الحفاظ على أكبر قدر ممكن من "المعلومات" (الممثلة بالتباين).
# 	# تعمل بإيجاد مجموعة جديدة من المحاور (المكونات الرئيسية) بحيث:
# 	# المحور الأول (PC1) يمسك بأكبر تباين ممكن.
# 	# المحور الثاني (PC2) يمسك بالثاني، وعمودي على الأول، وهكذا.
# 	# نحتفظ بأول k مكونًا يغطون 95% من التباين → نتجاهل الباقي (تعتبر "ضجيجًا").
#     pca = PCA(n_components=pca_energy)  # preserves 95% energy by default
    
#     P_reduced = pca.fit_transform(P_centered.T).T # (k, M), k << n_dim
#     EP = pca.components_  # (k, n_dim)
    
#     # Precompute coordinate to index map
#     style_index = {coord: idx for idx, coord in enumerate(style_centers)}

#     # Subsampled grid in input image (only valid centers)
#     i_vals = list(range(pad, H - pad, stride))
#     j_vals = list(range(pad, W - pad, stride))

#     # Helper: compute squared Euclidean distance in PCA space
#     def ssd_pca(i, j, k, l):
#         # Extract and flatten patch from input
#         patch_X = input_img[i - pad:i + pad + 1, j - pad:j + pad + 1].flatten()  # (n,)
#         # Project to PCA space
#         patch_X_centered = patch_X - mP.squeeze()  # mP.squeeze() → shape: (n,)
#         patch_X_reduced = EP @ patch_X_centered  # (k,) اسقاط :  تمثيل للـ patch في الفضاء المنخفض الأبعاد (k << n).

#         # Get index of (k,l)
#         idx = style_index.get((k, l), -1)
#         if idx == -1:
#             return np.inf

#         # Compute squared distance in reduced space
#         diff = patch_X_reduced - P_reduced[:, idx]
#         return np.dot(diff, diff)

#     # Helper: compute true SSD in RGB space (for final distances)
#     def ssd_original(i, j, k, l):
#         patch_X = input_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
#         patch_S = style_img[l - pad:l + pad + 1, k - pad:k + pad + 1]
#         return np.sum((patch_X - patch_S) ** 2)

#     # Initialize random matches
#     matches = {}
#     distances = {}
#     for i in i_vals:
#         for j in j_vals:
#             idx = rng.integers(0, M)
#             k, l = style_centers[idx]
#             matches[(i, j)] = np.array([k, l], dtype=np.float32)
#             distances[(i, j)] = ssd_original(i, j, k, l)

#     # PatchMatch iterations
#     for it in range(num_iters):
# 		#         التكرار 0: المعلومات تنتشر → و ↓
# 		# التكرار 1: المعلومات تنتشر ← و ↑
# 		#تغير اتجاه الانتشار في كل مرة
#         order_i = i_vals if it % 2 == 0 else reversed(i_vals)
#         order_j = j_vals if it % 2 == 0 else reversed(j_vals)

#         for i in order_i:
#             for j in order_j:
#                 best_dist_pca = ssd_pca(i, j, int(matches[(i, j)][0]), int(matches[(i, j)][1]))
#                 best_match = matches[(i, j)].copy()

#                 # Propagation
#                 neighbors = []
#                 if it % 2 == 0:
#                     if (i - stride, j) in matches:
#                         neighbors.append(matches[(i - stride, j)])
#                     if (i, j - stride) in matches:
#                         neighbors.append(matches[(i, j - stride)])
#                 else:
#                     if (i + stride, j) in matches:
#                         neighbors.append(matches[(i + stride, j)])
#                     if (i, j + stride) in matches:
#                         neighbors.append(matches[(i, j + stride)])

#                 # Evaluate neighbors in PCA space
#                 for cand in neighbors:
#                     k_cand, l_cand = np.clip(cand, [pad, pad], [Ws - pad - 1, Hs - pad - 1]).astype(int)
#                     d_pca = ssd_pca(i, j, k_cand, l_cand)
#                     if d_pca < best_dist_pca:
#                         best_dist_pca = d_pca
#                         best_match = np.array([k_cand, l_cand], dtype=np.float32)

#                 # Random search (in PCA space)
#                 alpha = 0.5
#                 radius = max(Hs, Ws)
#                 while radius >= 1:
#                     offset = rng.uniform(-radius, radius, size=2)
#                     cand = best_match + offset
#                     k_cand, l_cand = np.clip(cand, [pad, pad], [Ws - pad - 1, Hs - pad - 1]).astype(int)
#                     d_pca = ssd_pca(i, j, k_cand, l_cand)
#                     if d_pca < best_dist_pca:
#                         best_dist_pca = d_pca
#                         best_match = np.array([k_cand, l_cand], dtype=np.float32)
#                     radius = int(radius * alpha)

#                 # Update match and compute true SSD for final distance
#                 matches[(i, j)] = best_match
#                 k_final, l_final = best_match.astype(int)
#                 distances[(i, j)] = ssd_original(i, j, k_final, l_final)

#     return matches, distances

def patchmatch_ann_subsampled(
    input_img,
    style_img,
    patch_size=9,
    stride=5,
    num_iters=3,
    pca_energy=0.95,
    feature_space='rgb',            # NEW
    use_gaussian_weight=False,      # NEW
    init_nnf=None,                  # NEW
    random_seed=None                # NEW (match find_ann)
):
    rng = np.random.default_rng(random_seed)

    # --- 1. Convert to feature space (RGB → LAB) -----------------------------
    if feature_space == 'lab':
        if input_img.shape[2] != 3 or style_img.shape[2] != 3:
            raise ValueError("Lab conversion requires 3-channel RGB input.")
        input_f = rgb2lab(input_img.astype(np.float32) / 255.0)
        style_f = rgb2lab(style_img.astype(np.float32) / 255.0)
    else:
        input_f = input_img.astype(np.float32)
        style_f = style_img.astype(np.float32)

    H, W = input_f.shape[:2]
    Hs, Ws = style_f.shape[:2]
    pad = patch_size // 2
    C = input_f.shape[2] if input_f.ndim == 3 else 1

    # --- 2. Precompute style patches -----------------------------------------
    style_centers = []
    style_patches_flat = []
    for l in range(pad, Hs - pad):
        for k in range(pad, Ws - pad):
            p = style_f[l-pad:l+pad+1, k-pad:k+pad+1]
            if p.shape[:2] == (patch_size, patch_size):
                style_centers.append((k, l))
                style_patches_flat.append(p.flatten())

    if not style_patches_flat:
        raise ValueError("Style image too small for given patch_size")

    P = np.stack(style_patches_flat, axis=1)
    mP = np.mean(P, axis=1, keepdims=True)
    P_centered = P - mP

    # --- 3. PCA ---------------------------------------------------------------
    pca = PCA(n_components=pca_energy)
    P_reduced = pca.fit_transform(P_centered.T).T
    EP = pca.components_
    style_index = {coord: i for i, coord in enumerate(style_centers)}

    # --- 4. Gaussian weighting (optional) ------------------------------------
    if use_gaussian_weight:
        g = np.zeros((patch_size, patch_size))
        center = patch_size // 2
        for di in range(patch_size):
            for dj in range(patch_size):
                d2 = (di - center)**2 + (dj - center)**2
                g[di, dj] = np.exp(-d2 / (2 * (patch_size/6)**2))
        g = np.tile(g[:, :, None], (1, 1, C))
        g_flat = g.flatten()
    else:
        g_flat = None

    # --- 5. Subsample grid ----------------------------------------------------
    i_vals = list(range(pad, H - pad, stride))
    j_vals = list(range(pad, W - pad, stride))

    # --- 6. Initialize matches -------------------------------------------------
    matches = {}
    distances = {}

    for i in i_vals:
        for j in j_vals:

            if init_nnf is not None:
                # Use initial field
                k, l = init_nnf[i, j]
                if k < 0 or l < 0:
                    idx = rng.integers(0, len(style_centers))
                    k, l = style_centers[idx]
            else:
                # Random
                idx = rng.integers(0, len(style_centers))
                k, l = style_centers[idx]

            matches[(i, j)] = np.array([k, l], dtype=np.float32)

            # Store true distance
            patch_X = input_f[i-pad:i+pad+1, j-pad:j+pad+1]
            patch_S = style_f[l-pad:l+pad+1, k-pad:k+pad+1]
            dif = (patch_X - patch_S)

            if use_gaussian_weight:
                distances[(i, j)] = np.sum((dif**2).flatten() * g_flat)
            else:
                distances[(i, j)] = np.sum(dif**2)

    # --- 7. SSD helpers -------------------------------------------------------
    def ssd_pca(i, j, k, l):
        if k < pad or k >= Ws-pad or l < pad or l >= Hs-pad:
            return np.inf
        patch = input_f[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
        patch_centered = patch - mP.squeeze()
        reduced = EP @ patch_centered
        idx = style_index.get((k, l), -1)
        if idx == -1:
            return np.inf
        diff = reduced - P_reduced[:, idx]
        return np.dot(diff, diff)

    def ssd_true(i, j, k, l):
        pt = input_f[i-pad:i+pad+1, j-pad:j+pad+1]
        ps = style_f[l-pad:l+pad+1, k-pad:k+pad+1]
        diff = pt - ps
        if use_gaussian_weight:
            return np.sum((diff**2).flatten() * g_flat)
        return np.sum(diff**2)

    # --- 8. PatchMatch iterations --------------------------------------------
    for it in range(num_iters):
        order_i = i_vals if it % 2 == 0 else reversed(i_vals)
        order_j = j_vals if it % 2 == 0 else reversed(j_vals)

        for i in order_i:
            for j in order_j:
                best = matches[(i, j)].copy()
                best_dist = ssd_pca(i, j, int(best[0]), int(best[1]))

                # Propagation
                neighbors = []
                if it % 2 == 0:
                    if (i-stride, j) in matches:
                        neighbors.append(matches[(i-stride, j)])
                    if (i, j-stride) in matches:
                        neighbors.append(matches[(i, j-stride)])
                else:
                    if (i+stride, j) in matches:
                        neighbors.append(matches[(i+stride, j)])
                    if (i, j+stride) in matches:
                        neighbors.append(matches[(i, j+stride)])

                for cand in neighbors:
                    k_c, l_c = np.clip(cand,
                                      [pad, pad],
                                      [Ws-pad-1, Hs-pad-1]).astype(int)
                    d = ssd_pca(i, j, k_c, l_c)
                    if d < best_dist:
                        best_dist = d
                        best = np.array([k_c, l_c], dtype=np.float32)

                # Random search
                radius = max(Hs, Ws)
                alpha = 0.5
                while radius >= 1:
                    off = rng.uniform(-radius, radius, size=2)
                    cand = best + off
                    k_c, l_c = np.clip(cand,
                                      [pad, pad],
                                      [Ws-pad-1, Hs-pad-1]).astype(int)
                    d = ssd_pca(i, j, k_c, l_c)
                    if d < best_dist:
                        best_dist = d
                        best = np.array([k_c, l_c], dtype=np.float32)
                    radius = int(radius * alpha)

                matches[(i, j)] = best
                distances[(i, j)] = ssd_true(i, j,
                                             int(best[0]),
                                             int(best[1]))

    return matches, distances


def visualize_patchmatches(content_img, style_img, nnf, patch_size=9, num_samples=10):
    """
    Visualizes random content patches and their matched style patches.
    nnf can be:
        - dense array (H, W, 2)
        - sparse dict {(i,j): (x,y)}
    """

    H, W = content_img.shape[:2]
    pad = patch_size // 2

    # Normalize images if values >1
    if content_img.max() > 1: content_img = content_img / 255.0
    if style_img.max() > 1:   style_img   = style_img   / 255.0

    # --- Convert dict NNF → dense (H, W, 2) ---
    if isinstance(nnf, dict):
        dense_nnf = np.zeros((H, W, 2), dtype=np.float32)
        dense_nnf[:] = -1   # invalid
        for (i, j), v in nnf.items():
            dense_nnf[i, j] = v
        nnf = dense_nnf

    # --- Collect valid content coords ---
    valid_coords = []
    for i in range(pad, H - pad):
        for j in range(pad, W - pad):
            if nnf[i, j, 0] >= 0:
                valid_coords.append((i, j))

    if len(valid_coords) == 0:
        raise ValueError("No valid patch matches in NNF.")

    num_samples = min(num_samples, len(valid_coords))
    sampled = np.random.choice(len(valid_coords), num_samples, replace=False)

    # --- Plot ---
    plt.figure(figsize=(num_samples * 2, 4))

    for idx, s_idx in enumerate(sampled):
        i, j = valid_coords[s_idx]
        x, y = nnf[i, j].astype(int)

        # Extract patches
        content_patch = content_img[i-pad:i+pad+1, j-pad:j+pad+1]
        style_patch   = style_img[y-pad:y+pad+1, x-pad:x+pad+1]

        # Content patch
        plt.subplot(2, num_samples, idx + 1)
        plt.imshow(content_patch)
        plt.axis("off")
        if idx == 0: plt.title("Content Patches")

        # Style match
        plt.subplot(2, num_samples, idx + 1 + num_samples)
        plt.imshow(style_patch)
        plt.axis("off")
        if idx == 0: plt.title("Matched Style Patches")

    plt.show()



if __name__ == "__main__":
    # Load images
    content_img = load_image("Data/content/Mountain.jpg").astype(np.float32)
    style_img   = load_image("Data/style/scream.jpg").astype(np.float32)

    # Parameters
    patch_size = 9
    stride = 5
    num_iters = 2
    pca_energy = 0.95

    # Run PatchMatch
    nnf_dict, distances_dict= patchmatch_ann_subsampled(
        content_img,
        style_img,
        patch_size=patch_size,
        stride=stride,
        num_iters=num_iters,
        pca_energy=pca_energy,
        random_seed=5
    )
    # Dense NNF
    H, W = content_img.shape[:2]
    nnf = np.zeros((H, W, 2), dtype=np.float32)
    nnf[:] = -1
    for (i, j), v in nnf_dict.items():
        nnf[i, j] = v

    # Same for distances if needed
    distances = np.full((H, W), np.inf, dtype=np.float32)
    for (i, j), d in distances_dict.items():
        distances[i, j] = d

    print("NNF shape:", nnf.shape)
    print("Distance map shape:", distances.shape)

    # Inspect a few matches
    H, W = content_img.shape[:2]
    pad = patch_size // 2
    sample_coords = [(pad+10, pad+10), (pad+30, pad+20), (pad+50, pad+50)]
    for i,j in sample_coords:
        x, y = nnf[i,j].astype(int)
        print(f"Content patch at ({i},{j}) matched with style patch at ({x},{y}), distance={distances[i,j]:.2f}")

    # Visualize matches
    visualize_patchmatches(content_img, style_img, nnf, patch_size=patch_size, num_samples=10)