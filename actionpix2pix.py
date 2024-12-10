import torch
import numpy as np
from PIL import Image
import io
import os
import base64
from typing import Dict, Any
import logging
import traceback
import time
from dataclasses import dataclass
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from action_processing import ActionTokenizer


def load_or_create_unet():
    if not os.path.exists("../pt"):
        unet = UNet2DConditionModel.from_pretrained(
            "jackyk02/pix2pix", subfolder="unet", in_channels=8,
            safety_checker=None, from_flax=True
        ).to("cpu")
        unet.save_pretrained("../pt")
    return UNet2DConditionModel.from_pretrained(
        "../pt", safety_checker=None, torch_dtype=torch.float32,
        use_flash_attention=True
    ).to("cuda")

def load_models():
    unet = load_or_create_unet()
    vae = AutoencoderKL.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32)
    text_encoder = CLIPTextModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=torch.float32)
    tokenizer = CLIPTokenizer.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
    return unet, vae, text_encoder, tokenizer, scheduler

def create_pipeline(unet, vae, text_encoder, tokenizer, scheduler):
    pipe = StableDiffusionInstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False
    ).to("cuda")
    pipe.to(torch_dtype=torch.float32)
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def run_inference(pipe, prompt, image, num_inference_steps, guidance_scale, image_guidance_scale):
    start_time = time.time()
    
    output_image = pipe(
        prompt=prompt, image=image, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale
    ).images[0]
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    return output_image

class ActionPix2Pix:
    def __init__(self):
        unet, vae, text_encoder, tokenizer, scheduler = load_models()
        self.action_tokenizer = ActionTokenizer(tokenizer)
        self.pipe = create_pipeline(unet, vae, text_encoder, tokenizer, scheduler)

    def generate_image(self, image, action):
        prompt = f"what would it look like after taking the action {self.action_tokenizer(action)}?"
        output_image = run_inference(self.pipe, prompt, image, 50, 7.5, 2.5)
        return np.array(output_image)


# ap2p = ActionPix2Pix()
# image = Image.open("images/current_camera_image.jpg")
# action = np.array([0, 0.00784314, 0, 0, 0, 0.04705882, 0.99607843])

# generated_img = ap2p.generate_image(image, action)
# print(generated_img)
