from diffusers import StableDiffusionInpaintPipeline

import os
import time

import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from matplotlib import pyplot as plt

from GD_infer import eval

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


image = Image.open('val_stereo_nogt\Mirror1\camera_02\im2.png').convert("RGB")
print(image.size)
#image = image.resize((image.size[0]//5, image.size[1]//5))
mask = Image.fromarray(eval(['val_stereo_nogt\Mirror1\camera_02\im2.png'])[0]).convert("RGB")#.resize(image.size)
mask = pipe.mask_processor.blur(mask, blur_factor=33)

image = pipe(prompt="highly detailed, 8k, photorealistic, smooth , plain, furniture, matt finish, ((plastic))",
             negative_prompt = "reflective, transparent, brilliant, shinny, metallic, deformed, ugly",
             image=image, mask_image=mask,
             height=1024, width=1024, 
             strength  = 0.49, guidance_scale=12.0, 
             generator=torch.Generator(device="cuda").manual_seed(123)).images[0].resize(image.size)

plt.imshow(image)
plt.show()
plt.imshow(mask)
plt.show()