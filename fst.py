import torch
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(style_name="mosaic_n16"):
    path = os.path.join("models", f"{style_name}.pth")
    model = TransformerNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def apply_style_transfer(content_img, _, style_name="mosaic_n16"):
    model = load_model(style_name)
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor = preprocess(content_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).cpu().squeeze(0)
    post = transforms.Compose([
        transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                             std=[4.367, 4.464, 4.444]),
        transforms.ToPILImage()
    ])
    return post(out)

# Include TransformerNet, ConvLayer, ResidualBlock, UpsampleConvLayer here (same as previous)
