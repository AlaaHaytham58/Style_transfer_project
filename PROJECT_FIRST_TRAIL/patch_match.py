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
def patchmatch_ann_subsampled(
    input_img,
    style_img,
    patch_size=9,
    stride=5,
    num_iters=3,
    pca_energy=0.95,
    feature_space='rgb',
    use_gaussian_weight=False,
    init_nnf=None,
    random_seed=None
):
    rng = np.random.default_rng(random_seed)

    # --- Feature space ---
    if feature_space == 'lab':
        input_f = rgb2lab(input_img.astype(np.float32)/255.0)
        style_f = rgb2lab(style_img.astype(np.float32)/255.0)
    else:
        input_f = input_img.astype(np.float32)
        style_f = style_img.astype(np.float32)

    H, W = input_f.shape[:2]
    Hs, Ws = style_f.shape[:2]
    pad = patch_size // 2
    C = input_f.shape[2] if input_f.ndim==3 else 1

    # --- Precompute style patches ---
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

    pca = PCA(n_components=pca_energy)
    P_reduced = pca.fit_transform(P_centered.T).T
    EP = pca.components_
    style_index = {coord:i for i,coord in enumerate(style_centers)}

    # --- Gaussian weighting ---
    if use_gaussian_weight:
        g = np.zeros((patch_size, patch_size))
        center = patch_size // 2
        for di in range(patch_size):
            for dj in range(patch_size):
                d2 = (di-center)**2 + (dj-center)**2
                g[di,dj] = np.exp(-d2/(2*(patch_size/6)**2))
        g = np.tile(g[:,:,None], (1,1,C))
        g_flat = g.flatten()
    else:
        g_flat = None

    # --- Subsample grid ---
    i_vals = list(range(pad, H-pad, stride))
    j_vals = list(range(pad, W-pad, stride))

    # --- Initialize matches dict ---
    matches = {}
    distances = {}

    for i in i_vals:
        for j in j_vals:
            if init_nnf is not None:
                val = init_nnf.get((i,j), None)
                if val is None or val[0]<0 or val[1]<0:
                    k, l = style_centers[rng.integers(0,len(style_centers))]
                else:
                    k, l = int(val[0]), int(val[1])
            else:
                k, l = style_centers[rng.integers(0,len(style_centers))]

            matches[(i,j)] = np.array([k,l], dtype=np.float32)

            # Compute distance
            patch_X = input_f[i-pad:i+pad+1, j-pad:j+pad+1]
            patch_S = style_f[l-pad:l+pad+1, k-pad:k+pad+1]
            dif = patch_X - patch_S
            distances[(i,j)] = np.sum((dif**2).flatten()*g_flat) if use_gaussian_weight else np.sum(dif**2)

    # --- SSD helpers ---
    def ssd_pca(i,j,k,l):
        if k<pad or k>=Ws-pad or l<pad or l>=Hs-pad:
            return np.inf
        patch = input_f[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
        patch_centered = patch - mP.squeeze()
        reduced = EP @ patch_centered
        idx = style_index.get((k,l), -1)
        if idx==-1: return np.inf
        diff = reduced - P_reduced[:,idx]
        return np.dot(diff,diff)

    def ssd_true(i,j,k,l):
        pt = input_f[i-pad:i+pad+1, j-pad:j+pad+1]
        ps = style_f[l-pad:l+pad+1, k-pad:k+pad+1]
        diff = pt-ps
        return np.sum((diff**2).flatten()*g_flat) if use_gaussian_weight else np.sum(diff**2)

    # --- PatchMatch iterations ---
    for it in range(num_iters):
        order_i = i_vals if it%2==0 else reversed(i_vals)
        order_j = j_vals if it%2==0 else reversed(j_vals)

        for i in order_i:
            for j in order_j:
                best = matches[(i,j)].copy()
                best_dist = ssd_pca(i,j,int(best[0]),int(best[1]))

                # Propagation
                neighbors = []
                if it%2==0:
                    if (i-stride,j) in matches: neighbors.append(matches[(i-stride,j)])
                    if (i,j-stride) in matches: neighbors.append(matches[(i,j-stride)])
                else:
                    if (i+stride,j) in matches: neighbors.append(matches[(i+stride,j)])
                    if (i,j+stride) in matches: neighbors.append(matches[(i,j+stride)])

                for cand in neighbors:
                    k_c, l_c = np.clip(cand, [pad,pad],[Ws-pad-1,Hs-pad-1]).astype(int)
                    d = ssd_pca(i,j,k_c,l_c)
                    if d<best_dist:
                        best_dist=d
                        best=np.array([k_c,l_c],dtype=np.float32)

                # Random search
                radius = max(Hs, Ws)
                alpha = 0.5
                while radius>=1:
                    off = rng.uniform(-radius, radius, size=2)
                    cand = best + off
                    k_c, l_c = np.clip(cand, [pad,pad],[Ws-pad-1,Hs-pad-1]).astype(int)
                    d = ssd_pca(i,j,k_c,l_c)
                    if d<best_dist:
                        best_dist=d
                        best=np.array([k_c,l_c],dtype=np.float32)
                    radius = int(radius*alpha)

                matches[(i,j)] = best
                distances[(i,j)] = ssd_true(i,j,int(best[0]),int(best[1]))

    return matches, distances

  

# --- 7. Multi-scale wrapper ---
def find_ann_multiscale(target, source, patch_sizes, strides, **kwargs):
    nnf = None
    dist = None
    for ps,stride in zip(patch_sizes,strides):
        nnf, dist = patchmatch_ann_subsampled(
            target, source, patch_size=ps, stride=stride,
            init_nnf=nnf)
    return nnf, dist
 
 
def find_ann_multiscale_with_pyramid(
    target, source, patch_sizes_list, strides_list, num_scales=3,
    feature_space='rgb', pca_energy=0.95, random_seed=None, **kwargs
):
    """
    Multi-scale PatchMatch using dict-based NNF representation.

    Args:
        target: content image (H x W x C)
        source: style image (H x W x C)
        patch_sizes_list: list of lists of patch sizes per scale
        strides_list: list of lists of strides per scale
        num_scales: number of scales in pyramid
        feature_space: 'rgb' or 'lab'
        pca_energy: PCA variance to keep
        random_seed: for reproducibility

    Returns:
        nnf_up: dict mapping (i,j) -> [x,y] coordinates in source
        dist_curr: dict mapping (i,j) -> patch distance
    """
    rng = np.random.default_rng(random_seed)

    # --- Build Gaussian pyramids ---
    target_pyr = build_pyramid(target, num_scales)[::-1]  # coarse -> fine
    source_pyr = build_pyramid(source, num_scales)[::-1]

    X_current = None
    nnf_up = None

    for scale_idx in range(num_scales):
        curr_target = target_pyr[scale_idx]
        curr_source = source_pyr[scale_idx]
        H_curr, W_curr = curr_target.shape[:2]

        # --- Initialize or upsample X ---
        if scale_idx == 0:
            X_current = curr_target + rng.normal(0, 50, curr_target.shape)
        else:
            # Bilinear upsampling from previous estimate
            X_current = resize(
                X_current,
                (H_curr, W_curr, curr_target.shape[2]),
                order=1,
                anti_aliasing=False,
                preserve_range=True
            )

        # --- Upsample dict-based NNF from previous scale ---
        if nnf_up is not None:
            nnf_init = {}
            prev_source = source_pyr[scale_idx-1]
            prev_H, prev_W = prev_source.shape[:2]

            scale_x = curr_source.shape[1] / prev_W
            scale_y = curr_source.shape[0] / prev_H

            for (i,j), v in nnf_up.items():
                i_new = int(i * (H_curr / prev_H))
                j_new = int(j * (W_curr / prev_W))
                k_new = int(v[0] * scale_x)
                l_new = int(v[1] * scale_y)
                nnf_init[(i_new, j_new)] = np.array([k_new, l_new], dtype=np.float32)
        else:
            nnf_init = None

        # --- Run multi-patch-size PatchMatch ---
        patch_sizes = patch_sizes_list[scale_idx]
        strides = strides_list[scale_idx]

        nnf_curr, dist_curr = find_ann_multiscale(
            X_current, curr_source,
            patch_sizes=patch_sizes,
            strides=strides,
            init_nnf=nnf_init,
            feature_space=feature_space,
            pca_energy=pca_energy,
            random_seed=random_seed)

        # --- Prepare for next scale ---
        nnf_up = nnf_curr  # keep dict for upsampling

    return nnf_up, dist_curr


  
def visualize_nnf_matches(content_img, style_img, nnf, patch_size=9, num_samples=10):
    """
    Visualizes a few random patches in content image and their matched patches in style image.
    nnf: dict mapping (i,j) -> [x,y]
    """
    H, W = content_img.shape[:2]
    pad = patch_size // 2

    # Collect valid coordinates that exist in nnf
    valid_coords = [coord for coord in nnf.keys() if pad <= coord[0] < H - pad and pad <= coord[1] < W - pad]
    if len(valid_coords) < num_samples:
        num_samples = len(valid_coords)

    sampled_indices = np.random.choice(len(valid_coords), size=num_samples, replace=False)

    plt.figure(figsize=(num_samples * 2, 4))
    for idx, s_idx in enumerate(sampled_indices):
        i, j = valid_coords[s_idx]
        match = nnf.get((i, j))
        if match is None:
            continue
        x, y = match.astype(int)

        # Clamp coordinates to avoid out-of-bounds
        x = np.clip(x, pad, style_img.shape[1]-pad-1)
        y = np.clip(y, pad, style_img.shape[0]-pad-1)

        content_patch = content_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
        style_patch = style_img[y - pad:y + pad + 1, x - pad:x + pad + 1]

        # Content patch
        plt.subplot(2, num_samples, idx + 1)
        plt.imshow(np.clip(content_patch, 0, 1))
        plt.axis('off')
        if idx == 0:
            plt.title("Content Patches")

        # Matched style patch
        plt.subplot(2, num_samples, idx + 1 + num_samples)
        plt.imshow(np.clip(style_patch, 0, 1))
        plt.axis('off')
        if idx == 0:
            plt.title("Matched Style Patches")

    plt.show()


if __name__ == "__main__":
    # --- Load images ---
    content_img = load_image("Data/content/house.jpg")
    style_img = load_image("Data/style/scream.jpg")

    # --- Multi-scale PatchMatch parameters ---
    num_scales = 3
    patch_sizes_list = [
        [33, 21],  # coarse
        [13, 9],   # medium
        [9]        # fine
    ]
    strides_list = [
        [28, 18],  # coarse
        [8, 5],    # medium
        [5]        # fine
    ]


    #test withour pyramid 
    
    # # --- Run multi-scale PatchMatch ---
    # nnf, distances = find_ann_multiscale(
    #     content_img,
    #     style_img,
    #     patch_sizes_list=patch_sizes_list,
    #     strides_list=strides_list,
    #     num_scales=num_scales,
    #     feature_space='rgb',
    #     pca_energy=0.95,
    #     random_seed=42
    # )

    # print("NNF keys:", len(nnf))
    # print("Sample distances for some keys:")
    # sample_keys = list(nnf.keys())[:5]
    # for key in sample_keys:
    #     print(f"{key} -> {nnf[key]}")

    # # --- Inspect a few matches ---
    # pad = 9 // 2
    # H, W = content_img.shape[:2]
    # sample_coords = [(pad + 10, pad + 10), (pad + 30, pad + 20), (pad + 50, pad + 50)]
    # for i, j in sample_coords:
    #     match = nnf.get((i,j))
    #     if match is not None:
    #         x, y = match.astype(int)
    #         print(f"Content patch at ({i},{j}) matched with style patch at ({x},{y})")
    #     else:
    #         print(f"No NNF entry for ({i},{j})")

    # # --- Visualize matches ---
    # visualize_nnf_matches(content_img, style_img, nnf, patch_size=9, num_samples=10)


    #test with pyramid
    # --- Run multi-scale PatchMatch ---
    nnf, distances = find_ann_multiscale_with_pyramid(
        content_img,
        style_img,
        patch_sizes_list=patch_sizes_list,
        strides_list=strides_list,
        num_scales=num_scales,
        feature_space='rgb',
        pca_energy=0.95,
        random_seed=42
    )

    print("NNF keys:", len(nnf))
    print("Sample distances for some keys:")
    sample_keys = list(nnf.keys())[:5]
    for key in sample_keys:
        print(f"{key} -> {nnf[key]}")

    # --- Inspect a few matches ---
    pad = 9 // 2
    H, W = content_img.shape[:2]
    sample_coords = [(pad + 10, pad + 10), (pad + 30, pad + 20), (pad + 50, pad + 50)]
    for i, j in sample_coords:
        match = nnf.get((i,j))
        if match is not None:
            x, y = match.astype(int)
            print(f"Content patch at ({i},{j}) matched with style patch at ({x},{y})")
        else:
            print(f"No NNF entry for ({i},{j})")

    # --- Visualize matches ---
    visualize_nnf_matches(content_img, style_img, nnf, patch_size=9, num_samples=10)
