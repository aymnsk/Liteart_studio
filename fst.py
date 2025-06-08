import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Load VGG19 model for features
vgg = models.vgg19(pretrained=True).features.eval()

# Define transformation
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def image_to_tensor(image: Image.Image):
    return loader(image).unsqueeze(0)

def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Dummy FST-like effect using VGG feature fusion
def apply_style_transfer(content_img: Image.Image, style_img: Image.Image) -> Image.Image:
    content_tensor = image_to_tensor(content_img)
    style_tensor = image_to_tensor(style_img)

    with torch.no_grad():
        content_feat = vgg(content_tensor)[-1]
        style_feat = vgg(style_tensor)[-1]
        blended = content_tensor + 0.3 * (style_feat - content_feat)

    return tensor_to_image(blended)
