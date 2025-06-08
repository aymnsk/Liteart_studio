from PIL import Image, ImageEnhance, ImageFilter

def apply_deepdream(image):
    image = image.resize((512, 512))
    image = image.filter(ImageFilter.DETAIL)
    image = image.filter(ImageFilter.EDGE_ENHANCE)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    return image
