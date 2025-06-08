import streamlit as st
from PIL import Image
from fst import apply_style_transfer

st.title("LiteArt Studio - Neural Style Transfer")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Style options matching your .pth files (without extension)
style_options = ["candy", "mosaic", "rain_princess", "udnie"]

style_choice = st.selectbox("Choose a style:", style_options)

if uploaded_file is not None:
    content_img = Image.open(uploaded_file).convert("RGB")
    st.image(content_img, caption="Original Image", use_column_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Applying style transfer..."):
            output_img = apply_style_transfer(content_img, None, style_name=style_choice)
        st.image(output_img, caption=f"Styled Image - {style_choice}", use_column_width=True)
