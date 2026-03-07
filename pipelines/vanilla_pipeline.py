from pipelines.pipeline import InpaintingPipeLineScheme, InpaintPipelineInput
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import torch_utils as utils
import cv2 as cv


class InpaintPipelineVanilla(InpaintingPipeLineScheme):
    MODEL_ID = "Manojb/stable-diffusion-2-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SD_SCALE_FACTOR = 0.18215
    CFG_SCALE_FACTOR = 7.5

    def __init__(self, model_id=MODEL_ID, device=DEVICE, **kwargs):
        super().__init__(model_id, device, **kwargs)

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
        latent_height = mask_tensor.shape[2] // 8
        latent_width = mask_tensor.shape[3] // 8
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(latent_height, latent_width), mode='nearest')
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
        noise = torch.randn_like(init_latents)

        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0]) * (1 - mask_tensor))
                   + (noise * mask_tensor))

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
        return latents

    def mask_preprocessing(self, mask_image):
        return mask_image

    def image_preprocessing(self, real_image, mask_image):
        return real_image

    def preprocess(self, pipe_in: InpaintPipelineInput):
        pipe_in.mask_image = self.mask_preprocessing(pipe_in.mask_image)
        pipe_in.init_image = self.image_preprocessing(pipe_in.init_image, pipe_in.mask_image)
        return pipe_in

    def postprocess(self, image: Image.Image, mask_image: Image.Image):
        return image

    @torch.no_grad()
    def pipe(self, pipe_in: InpaintPipelineInput):
        pipe_in = self.preprocess(pipe_in)
        text_embeddings = self.encode_prompt(pipe_in.prompt, self.text_encoder, self.tokenizer)
        utils.clear_cache()
        latents = self.prepare_latents(pipe_in.init_image)
        utils.clear_cache()
        mask = self.prepare_mask_tensor(pipe_in.mask_image)
        utils.clear_cache()
        latents = self.denoise(text_embeddings, latents, mask)
        utils.clear_cache()
        final_image = self.decode_latents(latents)
        utils.clear_cache()
        return self.postprocess(final_image, mask)
