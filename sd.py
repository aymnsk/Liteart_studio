import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = None

def load_model():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            revision="fp16"
        )
        pipe = pipe.to("cpu")

def generate_image(prompt: str) -> Image.Image:
    load_model()
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    return image
