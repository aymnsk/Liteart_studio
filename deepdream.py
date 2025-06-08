from PIL import Image, ImageFilter, ImageEnhance

def apply_deepdream(img: Image.Image) -> Image.Image:
    # Simple fake "deepdream" effect by enhancing edges and color
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.8)
    return img
