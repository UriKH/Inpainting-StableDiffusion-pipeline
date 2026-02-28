from pipeline import SD2InpaintingPipeLineScheme, InpaintPipelineInput

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch_utils as utils


class InpaintPipeline(SD2InpaintingPipeLineScheme):
    MODEL_ID = "Manojb/stable-diffusion-2-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SD_SCALE_FACTOR = 0.18215

    def __init__(self, model_id=MODEL_ID, device=DEVICE):
        super().__init__(model_id, device)

    def encode_prompt(self, prompt, text_encoder, tokenizer):
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def prepare_latents(self, image):
        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            init_latents = self.vae.encode(image_tensor).latent_dist.sample()
            init_latents = self.SD_SCALE_FACTOR * init_latents
        return init_latents

    def prepare_mask_tensor(self, mask_image):
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        return mask_tensor

    def decode_latents(self, latents) -> Image.Image:
        with torch.no_grad():
            latents = 1 / self.SD_SCALE_FACTOR * latents
            image = self.vae.decode(latents).sample

        # Convert back to a PIL image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)
    
    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = torch.randn_like(init_latents)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
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
    
    @torch.no_grad()
    def pipe(self, pipe_in: InpaintPipelineInput):
        text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
        latents = self.prepare_latents(pipe_in.init_image)
        mask = self.prepare_mask_tensor(pipe_in.mask_image)
        latents = self.denoise(text_embeddings, latents, mask)
        final_image = self.decode_latents(latents)
        utils.clear_cache()
        return final_image
