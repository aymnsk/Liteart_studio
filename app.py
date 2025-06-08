import streamlit as st
from PIL import Image
import fst
import cartoon
import deepdream
# import sd  # 🚫 Disabled for now to avoid Huggingface errors

st.title("🎨 LiteArt Studio - Offline AI Art App")

st.sidebar.header("Upload Image & Choose Effect")
content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("Upload Style Image (for Fast Style Transfer only)", type=["jpg", "jpeg", "png"])

style_options = [
    "Fast Style Transfer",
    "DeepDream",
    "Cartoonize",
    "Pencil Sketch",
    # "Stable Diffusion Text2Image"  # 🚫 Temporarily Disabled
]
style_choice = st.sidebar.selectbox("Choose Style", style_options)

# Input check
if content_file is None and style_choice != "Stable Diffusion Text2Image":
    st.info("👆 Please upload a content image to begin.")
    st.stop()

if style_choice == "Fast Style Transfer" and style_file is None:
    st.warning("⚠️ Fast Style Transfer needs both a content and style image.")
    st.stop()

# Load image
content_image = Image.open(content_file).convert("RGB") if content_file else None
style_image = Image.open(style_file).convert("RGB") if style_file else None
output_image = None

with st.spinner("✨ Applying style..."):
    if style_choice == "Fast Style Transfer":
        output_image = fst.apply_style_transfer(content_image, style_image)
    elif style_choice == "Cartoonize":
        output_image = cartoon.apply_cartoon(content_image)
    elif style_choice == "Pencil Sketch":
        output_image = cartoon.apply_pencil_sketch(content_image)
    elif style_choice == "DeepDream":
        output_image = deepdream.apply_deepdream(content_image)
    # elif style_choice == "Stable Diffusion Text2Image":
    #     st.warning("🚫 Stable Diffusion temporarily disabled due to HuggingFace dependency issues.")
    #     st.stop()

if output_image:
    st.image(output_image, caption="🖼️ Output Image", use_column_width=True)
    output_image.save("output.png")
    st.success("✅ Saved as `output.png`")
