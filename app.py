import streamlit as st
from PIL import Image
import fst  # your updated Fast Style Transfer module
import cartoon  # your cartoon effect module
import deepdream  # your deepdream module

st.set_page_config(page_title="LiteArt Studio", layout="wide")

st.title("🎨 LiteArt Studio - Offline AI Art Studio")

# Sidebar - choose mode
mode = st.sidebar.selectbox(
    "Select an art style effect",
    ["Fast Style Transfer", "Cartoonize", "DeepDream", "Pencil Sketch"]
)

uploaded_file = st.file_uploader("Upload Content Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    content_image = Image.open(uploaded_file).convert("RGB")
    st.image(content_image, caption="Content Image", use_container_width=True)

    output_image = None

    if mode == "Fast Style Transfer":
        style_choice = st.sidebar.selectbox(
            "Select Style",
            ["mosaic_n16", "candy_n16", "rain_princess_n16", "udnie_n16"]
        )
        with st.spinner("Applying Fast Style Transfer..."):
            output_image = fst.apply_style_transfer(content_image, None, style_name=style_choice)

    elif mode == "Cartoonize":
        with st.spinner("Applying Cartoon Effect..."):
            output_image = cartoon.cartoonize_image(content_image)

    elif mode == "DeepDream":
        with st.spinner("Applying DeepDream..."):
            output_image = deepdream.apply_deepdream(content_image)

    elif mode == "Pencil Sketch":
        with st.spinner("Applying Pencil Sketch..."):
            # Simple OpenCV sketch effect
            import cv2
            import numpy as np
            img = cv2.cvtColor(np.array(content_image), cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_img = cv2.bitwise_not(gray_img)
            blur_img = cv2.GaussianBlur(inv_img, (21, 21), sigmaX=0, sigmaY=0)
            inv_blur = cv2.bitwise_not(blur_img)
            sketch_img = cv2.divide(gray_img, inv_blur, scale=256.0)
            output_image = Image.fromarray(cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2RGB))

    if output_image:
        st.image(output_image, caption="Output Image", use_container_width=True)
        output_image.save("output.png")
        with open("output.png", "rb") as f:
            st.download_button("Download Output Image", data=f, file_name="output.png")

else:
    st.info("Please upload a content image to start.")
