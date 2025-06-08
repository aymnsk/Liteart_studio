import streamlit as st
from PIL import Image
from fst import apply_style_transfer  # Make sure fst.py has your style transfer functions

st.set_page_config(page_title="LiteArt Studio - Neural Style Transfer", layout="centered")

st.title("LiteArt Studio 🎨 Neural Style Transfer")

# Available styles mapping
style_options = {
    "Mosaic": "mosaic",
    "Candy": "candy",
    "Rain Princess": "rain_princess",
    "Udnie": "udnie",
}

uploaded_file = st.file_uploader(
    "Upload an image (jpg, jpeg, png) to apply style transfer",
    type=["jpg", "jpeg", "png"]
)

style_choice = st.selectbox("Choose a style", list(style_options.keys()))

if uploaded_file is not None:
    try:
        content_img = Image.open(uploaded_file).convert("RGB")
        st.image(content_img, caption="Original Image", use_column_width=True)

        if st.button("Apply Style Transfer"):
            with st.spinner("Applying style... This may take a moment."):
                output_img = apply_style_transfer(content_img, style_name=style_options[style_choice])
            st.image(output_img, caption=f"Styled Image - {style_choice}", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to get started.")
