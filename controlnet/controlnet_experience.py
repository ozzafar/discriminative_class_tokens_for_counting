# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image, ImageDraw

prompt = "high resolution oranges"
negative_prompt = "low quality, bad quality, sketches"

# download an image
image = load_image(
    "C:\\Users\\ozzafar\\discriminative_class_tokens_for_counting\\controlnet\\5_dots.png"
)

# initialize the models and pipeline
controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float32
)
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float32
)
pipe.enable_model_cpu_offload()

# # Define the specific locations (coordinates) for oranges
# locations = [(50, 50), (100, 100), (150, 150), (200, 200), (250, 250),
#              (300, 300), (350, 350), (400, 400), (450, 450), (500, 500)]
#
# # Create a control image with the locations marked (e.g., black dots on a white background)
# control_image = Image.new('RGB', (512, 512), color='white')
# image = ImageDraw.Draw(control_image)
# for location in locations:
#     image.ellipse([location[0] - 10, location[1] - 10, location[0] + 10, location[1] + 10], fill='black')

# get canny image
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# generate image
image = pipe(
    prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, height=512, width=512
).images[0]

image.show()