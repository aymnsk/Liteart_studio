import streamlit as st
from PIL import Image
import fst
import cartoon
import deepdream

st.set_page_config(page_title="LiteArt Studio", layout="wide")
st.title("🎨 LiteArt Studio - AI Art Effects")

st.sidebar.header("Upload Image & Choose Effect")
content_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("Upload Style Image (for Fast Style Transfer)", type=["jpg", "jpeg", "png"])

style_options = [
    "Fast Style Transfer",
    "DeepDream",
    "Cartoonize",
    "Pencil Sketch"
]
style_choice = st.sidebar.selectbox("Choose Style", style_options)

if content_file is None:
    st.info("👆 Please upload a content image.")
    st.stop()

if style_choice == "Fast Style Transfer" and style_file is None:
    st.warning("⚠️ Style image is required for Fast Style Transfer.")
    st.stop()

content_image = Image.open(content_file).convert("RGB")
style_image = Image.open(style_file).convert("RGB") if style_file else None
output_image = None

with st.spinner("✨ Processing..."):
    if style_choice == "Fast Style Transfer":
        output_image = fst.apply_style_transfer(content_image, style_image)
    elif style_choice == "Cartoonize":
        output_image = cartoon.apply_cartoon(content_image)
    elif style_choice == "Pencil Sketch":
        output_image = cartoon.apply_pencil_sketch(content_image)
    elif style_choice == "DeepDream":
        output_image = deepdream.apply_deepdream(content_image)

if output_image:
    st.image(output_image, caption="🖼️ Output Image", use_container_width=True)
    output_image.save("output.png")
    st.success("✅ Saved as output.png")
