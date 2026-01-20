
---

# 🎨 Neural Style Transfer (NST) Project

🎓 **Field:** Computer & Communication Engineering

💻 **Domain:** Computer Vision & Deep Learning

📄 **Reference Paper:** [*A Neural Algorithm of Artistic Style* – Gatys et al. (2015)](https://arxiv.org/abs/1508.06576)

---

## 📌 Project Description

**Neural Style Transfer** is an optimization technique used to take two images—a **content image** and a **style reference image** (such as an artwork by Van Gogh)—and blend them together so the output image looks like the content image, but “painted” in the style of the reference.

This project utilizes a pre-trained **VGG-19 Convolutional Neural Network** to decouple the content and style representations of the input images.

---

## 🧠 Loss Functions (The Mathematics of Art)

The algorithm works by minimizing a multi-objective loss function. We define the total loss as a weighted sum:

### 1. Content Loss

To preserve the structure of the content image, we minimize the distance between the feature maps  (content) and  (generated) at a deep layer :


### 2. Style Loss

Style is captured using the **Gram Matrix** , which represents the correlations between different filter activations. The style loss is the difference between the style image Gram matrix  and the generated image Gram matrix :


### 3. Total Variation (TV) Loss

To ensure spatial continuity and reduce high-frequency artifacts (pixelation), we apply a denoising term:


---

## 🔄 Project Pipeline

| Stage | Process | Key Details |
| --- | --- | --- |
| **1. Preprocessing** | Image Resizing & Normalization | Resolution alignment and VGG-specific mean subtraction. |
| **2. Feature Extraction** | VGG-19 Forward Pass | Using `conv4_2` for content and `conv1_1` through `conv5_1` for style. |
| **3. Optimization** | Iterative Update | Gradient descent on the **pixel values** (not the model weights). |
| **4. Regularization** | TV Loss Application | Smoothing the generated image for a more natural look. |

---

## 🛠 Tech Stack & Requirements

* **Framework:** PyTorch / TensorFlow (Backend)
* **Libraries:** `numpy`, `opencv-python`, `matplotlib`, `scikit-image`, `tqdm`
* **Model:** VGG-19 (Pre-trained on ImageNet)

```bash
pip install numpy scipy opencv-python matplotlib scikit-image tqdm

```

---

## ▶️ How to Run

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/nst-project.git
cd nst-project

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Execute the Notebook:** Open `project-version2.ipynb` and run the cells. You can swap the `content_path` and `style_path` variables to test your own images.

---

## 🎥 Project Demo

https://drive.google.com/file/d/1JJwGCG2Cr3oNtUf47ZoNnI6OfnQ4cA1x/view?usp=sharing

---
---

## 🎓 Notes

* Project is **unsupervised**; no training data required
* Results may vary slightly due to randomized initialization
* Intended for academic experimentation in **computer vision & image processing**

```
---

## 🧠 Algorithm Structure

| Step | Description                                                           |
| ---- | --------------------------------------------------------------------- |
| 1    | Load content and style images; normalize resolution                   |
| 2    | Optional color transfer to align palettes                             |
| 3    | Build multi-scale Gaussian pyramids                                   |
| 4    | Extract overlapping patches of multiple sizes                         |
| 5    | Nearest-neighbor patch matching between synthesized and style patches |
| 6    | Aggregate style patches via weighted averaging                        |
| 7    | Blend reconstructed image with content to preserve structure          |
| 8    | Apply regularization/denoising                                        |
| 9    | Repeat across scales until convergence                                |
| 10   | Output final stylized image                                           |

---
