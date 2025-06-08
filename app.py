import streamlit as st
from PIL import Image
from fst import apply_style_transfer

st.title("LiteArt Studio - Neural Style Transfer")

style_options = {
    "mosaic": "mosaic",
    "candy": "candy",
    "rain_princess": "rain_princess",
    "udnie": "udnie",
}

uploaded_file = st.file_uploader("Upload an image to apply style transfer", type=["jpg", "jpeg", "png"])

style_choice = st.selectbox("Choose a style", list(style_options.keys()))

if uploaded_file is not None:
    content_img = Image.open(uploaded_file).convert("RGB")
    st.image(content_img, caption="Original Image", use_column_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Applying style..."):
            output_img = apply_style_transfer(content_img, style_name=style_options[style_choice])
            st.image(output_img, caption=f"Styled Image - {style_choice}", use_column_width=True)
