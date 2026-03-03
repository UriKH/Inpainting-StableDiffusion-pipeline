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


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self, blur_kernel=11, dilate_kernel=3, resampling_jumps=2):
        super().__init__()
        InpaintPipelineInput.MASK_DEFAULT_PREPROC_OP = partial(mask_op, dilate_kernel=dilate_kernel, kernel_size=blur_kernel)
        self.resampling_jumps = resampling_jumps
    
    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        for i, t in enumerate(self.scheduler.timesteps):

            # compute timestep
            if i < len(self.scheduler.timesteps) - 1:
                t_next = self.scheduler.timesteps[i + 1]
            else:
                t_next = torch.tensor([0], device=self.device)
            
            for j in range(self.resampling_jumps):
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # Step the scheduler
                latents_t_next = self.scheduler.step(noise_pred, t, latents).prev_sample


                # 1. Add noise to the original image matching the level we JUST stepped to
                noise = torch.randn_like(init_latents)
                noisy_background = self.scheduler.add_noise(init_latents, noise, t_next)
                
                # 2. Blend the accurately aligned latents
                blended_latents = (noisy_background * (1 - mask_tensor)) + (latents_t_next * mask_tensor)

                if j < self.resampling_jumps - 1:
                    beta_t = self.scheduler.betas[t].to(self.device)
                    jump_noise = torch.randn_like(blended_latents)
                    latents = torch.sqrt(1 - beta_t) * blended_latents + torch.sqrt(beta_t) * jump_noise
                else:
                    latents = blended_latents
        return latents
