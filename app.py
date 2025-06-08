import streamlit as st
from PIL import Image
import io

from fst import apply_style_transfer

# Set Streamlit page config
st.set_page_config(page_title="🎨 LiteArt Studio - Fast Style Transfer", layout="centered")

st.title("🎨 LiteArt Studio")
st.subheader("Transform your image with AI-powered Fast Style Transfer")

# Style options
style_options = {
    "Mosaic": "mosaic",
    "Candy": "candy",
    "Udnie": "udnie",
    "Rain Princess": "rain_princess"
}

style_choice = st.selectbox("Choose a Style", list(style_options.keys()))

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show original image
    content_img = Image.open(uploaded_file).convert("RGB")
    st.image(content_img, caption="Original Image", use_column_width=True)

    # Button to apply style
    if st.button("🎨 Stylize"):
        with st.spinner("Applying style... Please wait"):
            output_img = apply_style_transfer(content_img, None, style_name=f"{style_options[style_choice]}")
            st.image(output_img, caption=f"Stylized with {style_choice}", use_column_width=True)

            # Download button
            buf = io.BytesIO()
            output_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="📥 Download Stylized Image",
                               data=byte_im,
                               file_name=f"{style_choice.lower()}_styled.png",
                               mime="image/png")
