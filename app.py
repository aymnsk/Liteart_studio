import streamlit as st
from PIL import Image
from fst import apply_style_transfer

st.set_page_config(page_title="LiteArt Studio", layout="centered")

st.title("🎨 LiteArt Studio")
st.caption("Turn any photo into an artwork using Neural Style Transfer!")

# --- Available styles
style_options = {
    "🎭 Mosaic": "mosaic",
    "🍬 Candy": "candy",
    "🌧️ Rain Princess": "rain_princess",
    "🌀 Udnie": "udnie",
}

uploaded_file = st.file_uploader(
    "📤 Upload an image to stylize", type=["jpg", "jpeg", "png"]
)

style_choice = st.selectbox("🎨 Choose a style", list(style_options.keys()))

if uploaded_file:
    content_img = Image.open(uploaded_file).convert("RGB")
    st.image(content_img, caption="🖼️ Original Image", use_column_width=True)

    if st.button("✨ Apply Style"):
        with st.spinner("Styling... please wait..."):
            try:
                stylized_img = apply_style_transfer(content_img, style_name=style_options[style_choice])
                st.success("Done! 🎉")
                st.image(stylized_img, caption=f"🎨 Styled with {style_choice}", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
