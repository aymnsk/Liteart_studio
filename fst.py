import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image

def apply_fst(content_img):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # remove alpha channel
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor_img = transform(content_img).unsqueeze(0)

    # Load any lightweight style model – here using segmentation for dummy transform
    model = deeplabv3_resnet50(pretrained=True).eval()
    with torch.no_grad():
        _ = model(tensor_img)

    # In real implementation: apply style. Here, we just return resized input
    return content_img.resize((512, 512))
