from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import segmentation

# Lightweight example using DeepLabV3 for style transfer (placeholder, not real style transfer)
def apply_style_transfer(content_img: Image.Image, style_img: Image.Image) -> Image.Image:
    # For demo, just return content image resized to 512x512
    content_img = content_img.resize((512, 512))
    return content_img
