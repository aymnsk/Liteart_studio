import streamlit as st
from PIL import Image
import fst
import deepdream
import cartoon
import sd

st.set_page_config(page_title="🎨 LiteArt Studio", layout="centered")
st.title("🎨 LiteArt Studio - Offline AI Art Tools")

tab1, tab2 = st.tabs(["🖼️ Style Transfer & Effects", "🌌 Text to Image"])

with tab1:
    st.subheader("🖼 Upload Content & Style Image")
    content_file = st.file_uploader("Upload Content Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="content")
    style_file = st.file_uploader("Upload Style Image (Optional, JPG/PNG)", type=["jpg", "jpeg", "png"], key="style")

    style = st.selectbox("🧠 Choose a style to apply", [
        "Fast Style Transfer 🎨",
        "DeepDream 🌀",
        "Cartoonize 🖌️",
        "Pencil Sketch ✏️"
    ])

    if content_file:
        content_img = Image.open(content_file).convert('RGB')
        st.image(content_img, caption="🖼 Content Image", use_column_width=True)

        if style_file:
            style_img = Image.open(style_file).convert('RGB')
            st.image(style_img, caption="🎨 Style Image", use_column_width=True)

        if st.button("✨ Apply Style"):
            with st.spinner("Processing..."):
                if style == "Fast Style Transfer 🎨":
                    output = fst.apply_fst(content_img)
                elif style == "DeepDream 🌀":
                    output = deepdream.apply_deepdream(content_img)
                elif style == "Cartoonize 🖌️":
                    output = cartoon.cartoonize(content_img)
                elif style == "Pencil Sketch ✏️":
                    output = cartoon.pencil_sketch(content_img)

                st.image(output, caption="🖼 Output Image", use_column_width=True)
                output.save("output.png")
                st.success("Saved as output.png ✅")

with tab2:
    st.subheader("🌌 Generate Image from Prompt (Stable Diffusion)")
    prompt = st.text_input("Enter a text prompt", "a fantasy landscape in Van Gogh style")
    if st.button("🧠 Generate"):
        with st.spinner("Generating image (may take 1–2 mins on CPU)..."):
            pipe = sd.load_sd_model()
            image = sd.generate_image(pipe, prompt)
            st.image(image, caption="🌌 Generated Image", use_column_width=True)
