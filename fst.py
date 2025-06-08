import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [TransformerNet, ConvLayer, ResidualBlock, UpsampleConvLayer definitions unchanged]

# ... include your existing class definitions here ...

def load_model(style_name):
    # Map style choices to actual file names
    model_map = {
        "mosaic_n16": "mosaic.pth",
        "candy_n16": "candy.pth",
        "rain_princess_n16": "rain_princess.pth",
        "udnie_n16": "udnie.pth"
    }
    if style_name not in model_map:
        raise ValueError(f"Style '{style_name}' not supported.")
    
    model_path = os.path.join("models", model_map[style_name])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = TransformerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def apply_style_transfer(content_img, _, style_name="mosaic_n16"):
    model = load_model(style_name)
    
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3]),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    postprocess = transforms.Compose([
        transforms.Normalize([-2.118, -2.036, -1.804],
                             [4.367, 4.464, 4.444]),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.ToPILImage()
    ])
    
    content_tensor = preprocess(content_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(content_tensor).cpu().squeeze(0)
    return postprocess(output_tensor)
