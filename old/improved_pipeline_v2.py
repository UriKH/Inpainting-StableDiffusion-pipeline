from vanilla_pipeline import InpaintPipeline, InpaintPipelineInput
import torch
import numpy as np
import cv2 as cv
from PIL import Image
from functools import partial


def mask_op(image, dilate_kernel, kernel_size):
    mask_np = np.array(image)
    _, mask_np = cv.threshold(mask_np, 127, 255, cv.THRESH_BINARY)
    if dilate_kernel != 1:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask_np = cv.dilate(mask_np, kernel, iterations=1)
    if kernel_size != 1:
        mask_np = cv.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
    return Image.fromarray(mask_np).resize((64, 64), Image.LANCZOS)


class ImprovedInpaintPipelineV2(InpaintPipeline):
    def __init__(self, blur_kernel=11, dilate_kernel=3, resampling_jumps=2):
        super().__init__()
        InpaintPipelineInput.MASK_DEFAULT_PREPROC_OP = partial(mask_op, dilate_kernel=dilate_kernel, kernel_size=blur_kernel)
        self.resampling_jumps = resampling_jumps
    
    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        # ==========================================
        # PHASE 1: Vanilla Inpainting (The "Paste")
        # ==========================================
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand and predict noise
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform CFG guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # Step the scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Vanilla Mask Blending
            if i < len(self.scheduler.timesteps) - 1:
                t_next = self.scheduler.timesteps[i + 1]
            else:
                t_next = torch.tensor([0], device=self.device)

            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t_next)
            latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)

        # ==========================================
        # PHASE 2: Iterative Refinement (The "Smudge")
        # ==========================================

        import torchvision.transforms.functional as TF
        # Apply a Gaussian blur directly to the PyTorch mask tensor
        soft_mask_tensor = TF.gaussian_blur(mask_tensor, kernel_size=7, sigma=2.0)
        
        refine_start_step = int(num_inference_steps * 0.85) # Reduced strength!
        t_refine = self.scheduler.timesteps[refine_start_step]
        
        refine_noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, refine_noise, t_refine)

        for i, t in enumerate(self.scheduler.timesteps[refine_start_step:]):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Use the SOFT mask to protect the far background from turning into cupcakes!
            noise = torch.randn_like(init_latents)
            if i < len(self.scheduler.timesteps[refine_start_step:]) - 1:
                 t_next = self.scheduler.timesteps[refine_start_step + i + 1]
            else:
                 t_next = torch.tensor([0], device=self.device)
                 
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t_next)
            latents = (noisy_init_latents * (1 - soft_mask_tensor)) + (latents * soft_mask_tensor)
        
        return latents
