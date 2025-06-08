from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

pipe = None

def load_sd_model():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            revision="fp32",
        )
        pipe = pipe.to("cpu")
    return pipe

def generate_image(pipe, prompt):
    image = pipe(prompt).images[0]
    return image
