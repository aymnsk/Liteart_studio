import streamlit as st
from PIL import Image
import fst
import cartoon
import deepdream

st.set_page_config(page_title="LiteArt Studio", layout="centered")
st.title("🎨 LiteArt Studio")

style_choices = ["mosaic_n16", "candy_n16", "rain_princess_n16", "udnie_n16"]
mode = st.sidebar.selectbox("Mode", ["Fast Style Transfer"] + ["Cartoonize", "DeepDream"])
style_choice = None
if mode == "Fast Style Transfer":
    style_choice = st.sidebar.selectbox("Style", style_choices)

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)
    output = None

    if st.button("Run"):
        with st.spinner("Processing..."):
            if mode == "Fast Style Transfer":
                try:
                    output = fst.apply_style_transfer(img, None, style_name=style_choice)
                except Exception as e:
                    st.error(f"Error: {e}")
            elif mode == "Cartoonize":
                output = cartoon.apply_cartoon(img)
            elif mode == "DeepDream":
                output = deepdream.apply_deepdream(img)
            
            if output:
                st.image(output, caption="Output", use_container_width=True)
                st.download_button("Download", data=output.tobytes(), file_name="output.png", mime="image/png")
