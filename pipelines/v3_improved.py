from v2_improved import ImprovedInpaintPipelineV2
from cross_attention import MaskedCrossAttnProcessor
from pipeline import InpaintPipelineInput

import torch
from PIL import Image
import cv2 as cv
import numpy as np
import math

import torchvision.transforms.functional as TF
import torch.nn.functional as F



class ImprovedInpaintPipelineV3(ImprovedInpaintPipelineV2):
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
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        noise = torch.randn_like(init_latents)

        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0]) * (1 - mask_tensor))
                   + (noise * mask_tensor))

        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        
        # Inject the soft mask (while the latent blending loop still uses the strict binary mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask)

        try:
            for i, t in enumerate(self.scheduler.timesteps):
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)

                # Step the scheduler
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # Update timesteps and add noise
                if i < len(self.scheduler.timesteps) - 1:
                    t_next = self.scheduler.timesteps[i + 1]

                    # Add noise to the original image matching the level we JUST stepped to
                    noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, noise, t_next)
                else:
                    # At the final step, the known background should be perfectly clean
                    known_background = init_latents

                # Blend the accurately aligned latents
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents