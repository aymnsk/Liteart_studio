import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = None

def load_model():
    global pipe
    if pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)

def generate_image(prompt: str) -> Image.Image:
    load_model()
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    return image
