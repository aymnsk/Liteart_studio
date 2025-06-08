import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

# Preprocess and postprocess functions
def image_to_tensor(image, max_size=512):
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # Remove alpha if present
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    unloader = transforms.Compose([
        transforms.Normalize(mean=[-2.12, -2.04, -1.80],
                             std=[4.37, 4.46, 4.44]),
        transforms.ToPILImage()
    ])
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    return image

# Define the Style Transfer Module
def apply_style_transfer(content_image, style_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model (VGG19)
    vgg = models.vgg19(pretrained=True).features[:21].to(device).eval()

    for param in vgg.parameters():
        param.requires_grad = False

    content = image_to_tensor(content_image).to(device)
    style = image_to_tensor(style_image).to(device)

    generated = content.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS([generated])

    style_features = get_features(style, vgg)
    content_features = get_features(content, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    style_weight = 1e6
    content_weight = 1e0

    run = [0]
    while run[0] <= 300:

        def closure():
            generated.data.clamp_(0, 1)
            optimizer.zero_grad()
            gen_features = get_features(generated, vgg)

            content_loss = F.mse_loss(gen_features['conv4_2'], content_features['conv4_2'])
            style_loss = 0

            for layer in style_grams:
                gen_gram = gram_matrix(gen_features[layer])
                style_gram = style_grams[layer]
                style_loss += F.mse_loss(gen_gram, style_gram)

            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            run[0] += 1
            return total_loss

        optimizer.step(closure)

    output_img = generated.data.clamp_(0, 1)
    return tensor_to_image(output_img)

# Helper functions
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)
