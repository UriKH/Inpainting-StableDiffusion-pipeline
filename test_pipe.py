import torch
import numpy as np
from PIL import Image

# (Assuming you paste your load_sd2_components function here)
# ...

import dataclasses
import os

import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)

from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
from dataclasses import dataclass


def load_sd2_components(model_path, device="cuda"):
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float32).to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                     torch_dtype=torch.float32).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch.float32)
    print(f'{"=" * 10} Components Loaded {"=" * 10}')
    return vae, unet, text_encoder, tokenizer, scheduler



device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Manojb/stable-diffusion-2-base" 

# 1. Load Components
vae, unet, text_encoder, tokenizer, scheduler = load_sd2_components(model_id, device=device)

# 2. Inputs
prompt = "a bright red apple"
init_image = Image.open("./data/ours/masked/dog.png").convert("RGB").resize((512, 512))
mask_image = Image.open("./data/ours/masked/dog.mask.png").convert("L").resize((64, 64)) # Mask must match latent size (512/8 = 64)

# 3. Encode Prompt (with Classifier-Free Guidance)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# 4. Prepare Image Latents
image_np = np.array(init_image).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    init_latents = vae.encode(image_tensor).latent_dist.sample()
    init_latents = 0.18215 * init_latents  # Magic scaling factor for SD

# 5. Prepare Mask Tensor
mask_np = np.array(mask_image).astype(np.float32) / 255.0
# Ensure mask is 1 for the hole (where we generate) and 0 for background
mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

# 6. Set up the Denoising Loop
num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

# Start with pure random noise
latents = torch.randn_like(init_latents)

print("Starting the custom denoising loop...")
for i, t in enumerate(timesteps):
    # Expand latents for classifier free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Predict noise
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

    # Step the scheduler
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    # --- THE VANILLA INPAINTING MAGIC HAPPENS HERE ---
    # 1. Add the correct amount of noise to the original image for this specific timestep
    noise = torch.randn_like(init_latents)
    noisy_init_latents = scheduler.add_noise(init_latents, noise, t)
    
    # 2. Force the background to stay true to the original image, while keeping the AI's generation inside the mask
    latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)
    # ------------------------------------------------

# 7. Decode Latents back to Image
with torch.no_grad():
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

# Convert back to a PIL image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = (image * 255).round().astype("uint8")

final_image = Image.fromarray(image)
final_image.save("custom_loop_output.png")
print("Done! Saved to custom_loop_output.png")
