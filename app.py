import streamlit as st
import numpy as np
from engine.pipeline import run_style_transfer
from engine.metrics import compute_metrics
from PIL import Image
import io

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(layout="wide")
st.title("🎨 Patch-Based Style Transfer")

# ======================
# SIDEBAR PARAMETERS
# ======================
st.sidebar.header("Style Transfer Parameters")

patch_sizes = st.sidebar.multiselect(
    "Patch Sizes",
    [7, 9, 13, 17, 21, 33],
    default=[33, 21, 13, 9]
)

gaps = st.sidebar.multiselect(
    "Subsampling Gaps",
    [3, 5, 8, 18, 28],
    default=[28, 18, 8, 5]
)

r = st.sidebar.slider("Robust Norm (r)", 0.3, 1.0, 0.8, 0.01)
irls = st.sidebar.slider("IRLS Iterations", 1, 20, 10)
em = st.sidebar.slider("EM Iterations", 1, 5, 3)
levels = st.sidebar.slider("Pyramid Levels", 1, 5, 3)
max_size = st.sidebar.slider("Max Image Size", 256, 768, 400, 32)

st.sidebar.markdown("---")
st.sidebar.header("Post-Processing")

brightness = st.sidebar.slider(
    "Brightness",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05
)

# ======================
# IMAGE INPUT
# ======================
st.header("Input Images")

col1, col2, col3 = st.columns(3)

with col1:
    content_file = st.file_uploader("Content Image", ["jpg", "png", "jpeg"])

with col2:
    style_file = st.file_uploader("Style Image", ["jpg", "png", "jpeg"])

with col3:
    dnn_file = st.file_uploader(
        "Pretrained DNN Output (for comparison)",
        ["jpg", "png", "jpeg"]
    )

# ======================
# HELPERS
# ======================
def apply_brightness(img, factor):
    img = img.astype(np.float32) / 255.0
    img = np.clip(img * factor, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

def numpy_image_to_bytes(img_np):
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================
# RUN
# ======================
if st.button("🚀 Run Style Transfer") and content_file and style_file:
    content_img = np.array(Image.open(content_file).convert("RGB"))
    style_img = np.array(Image.open(style_file).convert("RGB"))

    with st.spinner("Processing style transfer..."):
        output, metrics_ours = run_style_transfer(
            content_img,
            style_img,
            patch_sizes=patch_sizes,
            gaps=gaps,
            r_robust=r,
            irls_iterations=irls,
            em_iterations=em,
            num_levels=levels,
            max_size=max_size
        )

    # Post-processing
    output = apply_brightness(output, brightness)

    # ======================
    # DISPLAY RESULTS
    # ======================
    st.header("Results")

    if dnn_file is not None:
        dnn_output = np.array(Image.open(dnn_file).convert("RGB"))

        # Compute metrics for DNN output (same metrics!)
        metrics_dnn = compute_metrics(
            content_img.astype(np.uint8),
            style_img.astype(np.uint8),
            dnn_output.astype(np.uint8)
        )

        colA, colB = st.columns(2)

        with colA:
            st.image(output, caption="Our Patch-Based System", use_container_width=True)

        with colB:
            st.image(dnn_output, caption="Pretrained DNN Output", use_container_width=True)

        # ======================
        # COMPARISON TABLE
        # ======================
        st.subheader("📊 Quantitative Comparison")

        st.table({
            "Metric": list(metrics_ours.keys()),
            "Our System": list(metrics_ours.values()),
            "Pretrained DNN": list(metrics_dnn.values())
        })

    else:
        st.image(output, caption="Stylized Output", use_container_width=True)
        st.info("Upload a pretrained DNN output image to enable quantitative comparison.")

    # ======================
    # DOWNLOAD
    # ======================
    img_bytes = numpy_image_to_bytes(output)

    st.download_button(
        label="💾 Download Result Image",
        data=img_bytes,
        file_name="stylized_result.png",
        mime="image/png"
    )
