from vanilla_pipeline import InpaintPipeline
import torch
import numpy as np
import cv2 as cv


class ImprovedInpaintPipeline(InpaintPipeline):
    def __init__(self):
        super().__init__()

    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps)
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
            # 1. Add the correct amount of noise to the original image for this specific timestep
            noise = torch.randn_like(init_latents)
            noisy_init_latents = self.scheduler.add_noise(init_latents, noise, t)

            # 2. Force the background to stay true to the original image, while keeping the AI's generation inside the mask
            latents = (noisy_init_latents * (1 - mask_tensor)) + (latents * mask_tensor)
        return latents

    def prepare_mask_tensor(self, mask_image):
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask_np = cv.blur(mask_np, (3, 3))
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return mask_tensor
