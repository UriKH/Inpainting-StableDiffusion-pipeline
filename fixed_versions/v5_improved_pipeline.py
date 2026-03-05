from v3_improved_pipeline import MaskedCrossAttnProcessor, ImprovedInpaintPipeline as InpaintPipeline
from pipeline import InpaintPipelineInput
import torch
from PIL import Image
import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
import math
import torchvision.transforms.functional as TF

class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self, jump_length=10, jump_n_sample=2):
        super().__init__()
        # Store resampling hyperparameters as instance variables 
        # to preserve the base pipe() signature.
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
    
    def _create_soft_mask(self, mask_tensor, dilation_kernel=5, blur_kernel=15, sigma=5.0):
        """
        Applies dilation and Gaussian blur to the binary mask tensor.
        mask_tensor expected shape: (1, 1, H, W)
        """
        # 1. Dilation using Max Pooling (expands the 1s outward)
        pad = dilation_kernel // 2
        dilated_mask = F.max_pool2d(mask_tensor, kernel_size=dilation_kernel, stride=1, padding=pad)
        
        # 2. Gaussian Blur to create the soft gradient transition
        soft_mask = TF.gaussian_blur(dilated_mask, kernel_size=[blur_kernel, blur_kernel], sigma=[sigma, sigma])
        
        # Ensure values are strictly bounded mathematically
        return torch.clamp(soft_mask, 0.0, 1.0)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        """Overrides the base denoise method to include time-travel resampling."""
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 1. Generate the custom time-travel schedule
        schedule_indices = self._get_repaint_schedule(num_inference_steps)
        
        # 2. Initial Setup
        noise = torch.randn_like(init_latents)
        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor))
                   + (noise * mask_tensor))
        
        # Inside denoise()
        _, _, latent_h, latent_w = init_latents.shape
        
        # Generate the soft mask for the Cross-Attention processor
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        
        # Inject the soft mask (while the latent blending loop still uses the strict binary mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = self.scheduler.timesteps[step_index]
    
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
    
                # Step the scheduler to get t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # 3. Handle Time-Travel and Background Context
                is_last_step = (idx == len(schedule_indices) - 1)
            
                if not is_last_step:
                    next_step_index = schedule_indices[idx + 1]
                    t_next = self.scheduler.timesteps[next_step_index]
                
                    is_jump_backward = next_step_index < step_index
                
                    if is_jump_backward:
                        # Time travel! Apply the DDPM forward equation to inject noise back in
                        beta = self.scheduler.betas[t_next].to(self.device)
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * noise
                    
                    # Generate fresh stochastic background context for the upcoming step
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    # Final cleanup step resolving to sharpness
                    known_background = init_latents
    
                # 4. Blend the accurately aligned latents
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents
