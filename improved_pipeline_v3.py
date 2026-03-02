from vanilla_pipeline import InpaintPipeline
import torch
import numpy as np
import cv2 as cv
from PIL import Image

# 1. Strict Binary Dilation Mask (No blur here, we want strict latent boundaries!)
def mask_op(image):
    mask_np = np.array(image)
    _, mask_np = cv.threshold(mask_np, 127, 255, cv.THRESH_BINARY)
    
    # Large dilation to give the AI a canvas to cast shadows
    dilate_kernel = np.ones((15, 15), np.uint8)
    mask_np = cv.dilate(mask_np, dilate_kernel, iterations=1)
    
    return Image.fromarray(mask_np).resize((64, 64), Image.LANCZOS)

class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()
        # Override the mask operation specifically for the improved pipeline
        self.__class__.MASK_DEFAULT_PREPROC_OP = staticmethod(mask_op)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        # 2. Strict Vanilla Latent Blending
        # This guarantees the background (onions) stays safe, while the AI draws the cupcake
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if i < len(self.scheduler.timesteps) - 1:
                t_next = self.scheduler.timesteps[i + 1]
            else:
                t_next = torch.tensor([0], device=self.device)

            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t_next)
            
            # Force background to stay true
            latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)
            
        return latents

    @torch.no_grad()
    def pipe(self, pipe_in):
        # 1. Run the standard generation
        text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
        latents = self.prepare_latents(pipe_in.init_image)
        mask = self.prepare_mask_tensor(pipe_in.mask_image)
        latents = self.denoise(text_embeddings, latents, mask)
        
        # This image will have the "sticker" harsh edges from the vanilla loop
        ai_generated_image = self.decode_latents(latents)

        # ==========================================
        # 3. POISSON IMAGE EDITING (The Magic Polish)
        # ==========================================
        # Convert PIL images to OpenCV numpy arrays (RGB)
        src = np.array(ai_generated_image)       # The image with the new AI object
        dst = np.array(pipe_in.init_image)       # The completely untouched original image
        
        # We need a 512x512 mask for OpenCV. We upscale the 64x64 latent mask.
        mask_512 = pipe_in.mask_image.resize((512, 512), Image.NEAREST)
        mask_cv = np.array(mask_512)
        
        # OpenCV seamlessClone requires the mask to be 255 (white) for the object being pasted
        # and it needs to be a 3-channel array (or 1-channel 8-bit)
        if len(mask_cv.shape) == 2:
            mask_cv = cv.cvtColor(mask_cv, cv.COLOR_GRAY2RGB)

        # Find the center point of the bounding box of our mask
        # seamlessClone needs to know roughly where to paste the object
        x, y, w, h = cv.boundingRect(mask_cv[:, :, 0])
        
        # Fallback just in case the mask is entirely empty
        if w == 0 or h == 0:
            return ai_generated_image
            
        center = (x + w // 2, y + h // 2)

        # Apply Poisson Blending! 
        # NORMAL_CLONE mathematically adjusts the colors and gradients of the AI object to match the background lighting.
        blended_np = cv.seamlessClone(src, dst, mask_cv, center, cv.NORMAL_CLONE)

        return Image.fromarray(blended_np)
