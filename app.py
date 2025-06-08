import streamlit as st
from PIL import Image
import fst  # your style transfer module
import os

st.set_page_config(page_title="🎨 LiteArt Studio", layout="centered")
st.title("🎨 LiteArt Studio - Fast Style Transfer")

# Sidebar style choices
style_options = ["mosaic_n16", "udnie_n16", "candy_n16", "rain_princess_n16"]
style_choice = st.sidebar.selectbox("Choose a Style", style_options)

uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    content_image = Image.open(uploaded_image).convert("RGB")
    st.image(content_image, caption="🖼️ Original Image", use_column_width=True)

    if st.button("✨ Stylize Image"):
        with st.spinner("🪄 Applying style... please wait..."):
            try:
                output_image = fst.apply_style_transfer(content_image, None, style_name=style_choice)
                st.success("✅ Style transfer complete!")
                st.image(output_image, caption=f"🎨 Stylized ({style_choice})", use_column_width=True)

                # Save the output locally (optional)
                output_path = f"output_{style_choice}.jpg"
                output_image.save(output_path)
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Stylized Image",
                        data=file,
                        file_name=output_path,
                        mime="image/jpeg"
                    )

            except FileNotFoundError as e:
                st.error(f"🚫 Error: {e}")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
