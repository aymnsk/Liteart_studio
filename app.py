import streamlit as st
from PIL import Image
import fst
import cartoon
import deepdream
import sd

st.title("LiteArt Studio - Offline AI Art Studio")

st.sidebar.header("Upload Images & Choose Style")

content_file = st.sidebar.file_uploader("Upload Content Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader("Upload Style Image (Optional, JPG/PNG)", type=["jpg", "jpeg", "png"])

style_options = ["Fast Style Transfer", "DeepDream", "Cartoonize", "Pencil Sketch", "Stable Diffusion Text2Image"]
style_choice = st.sidebar.selectbox("Select Style / Effect", style_options)

if style_choice == "Stable Diffusion Text2Image":
    prompt = st.text_input("Enter text prompt for Stable Diffusion")

if content_file or style_choice == "Stable Diffusion Text2Image":

    if style_choice != "Stable Diffusion Text2Image":
        if content_file is None:
            st.warning("Please upload a content image to proceed.")
            st.stop()
        content_image = Image.open(content_file).convert("RGB")
    else:
        content_image = None

    style_image = None
    if style_file:
        style_image = Image.open(style_file).convert("RGB")

    output_image = None
    with st.spinner("Processing..."):
        if style_choice == "Fast Style Transfer":
            if style_image is None:
                st.warning("Please upload a style image for Fast Style Transfer.")
                st.stop()
            output_image = fst.apply_style_transfer(content_image, style_image)
        elif style_choice == "DeepDream":
            output_image = deepdream.apply_deepdream(content_image)
        elif style_choice == "Cartoonize":
            output_image = cartoon.apply_cartoon(content_image)
        elif style_choice == "Pencil Sketch":
            output_image = cartoon.apply_pencil_sketch(content_image)
        elif style_choice == "Stable Diffusion Text2Image":
            if not prompt:
                st.warning("Please enter a prompt for Stable Diffusion.")
                st.stop()
            output_image = sd.generate_image(prompt)

    if output_image:
        st.image(output_image, caption="Output Image", use_column_width=True)
        output_image.save("output.png")
        st.success("Image saved as output.png")

else:
    st.info("Upload images or enter prompt to get started.")
