from vanilla_pipeline import InpaintPipeline
import torch
import numpy as np
import cv2 as cv


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # Step the scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # --- THE VANILLA INPAINTING MAGIC HAPPENS HERE ---
            if i < len(self.scheduler.timesteps) - 1:
                t_next = self.scheduler.timesteps[i + 1]
            else:
                t_next = torch.tensor([0], device=self.device)

            # 1. Add noise to the original image matching the level we JUST stepped to
            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t_next)

            # 2. Blend the accurately aligned latents
            latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)
        return latents
            
    def prepare_mask_tensor(self, mask_image):
        mask_np = np.array(mask_image)
        kernel = np.ones((3, 3), np.uint8)
        mask_np = cv.dilate(mask_np, kernel, iterations=1)
        mask_np = cv.GaussianBlur(mask_np, (5, 5), 0)

        mask_np = mask_np.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return mask_tensor
