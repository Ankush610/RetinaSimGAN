# ==============================

# from diffusers import DiffusionPipeline

# pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt).images[0]


#============================

# import clip
# import os
# import torch

# # Create the cache directory if it doesn't exist
# os.makedirs(os.path.expanduser("~/.cache/clip"), exist_ok=True)

# # Download the model
# model, _ = clip.load("ViT-B/32", device="cpu")

# print(f"Model successfully downloaded to: {os.path.expanduser('~/.cache/clip')}")


#==============================================

import torch
import torchvision.models as models
import os

# Create the cache directory structure
os.makedirs(os.path.expanduser("~/.cache/torch/hub/checkpoints"), exist_ok=True)

# Download VGG16 model
vgg16 = models.vgg16(pretrained=True)
print("VGG16 model successfully downloaded")

#=========================================================================

import torch
torch.hub.download_url_to_file(
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
    os.path.expanduser("~/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth")
)

#=========================================================================

# /home/omjadhav/ankush/img2img-turbo/output/checkpoints/model_1001.pkl