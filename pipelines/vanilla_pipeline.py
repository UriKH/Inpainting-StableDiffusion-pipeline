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
    CFG_SCALE_FACTOR = 7.5

    def __init__(self, model_id=MODEL_ID, device=DEVICE, **kwargs):
        super().__init__(model_id, device, **kwargs)

    def encode_prompt(self, prompt: str, text_encoder, tokenizer):
        """
        Create the prompt embeddings and the unconditional prompt embeddings.
        :param prompt: The prompt to encode
        :param text_encoder: The text encoder model
        :param tokenizer: The tokenizer model
        :return: The prompt embeddings [uncond, text]
        """
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def prepare_latents(self, image: Image.Image):
        """
        Prepare the latent variables for the inpainting process.
        :param image: The image to prepare the latents for
        :return: The prepared latents
        (This function was implemented with the help of AI)
        """
        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            init_latents = self.vae.encode(image_tensor).latent_dist.sample()
            init_latents = self.vae.config.scaling_factor * init_latents
        return init_latents

    def prepare_mask_tensor(self, mask_image: Image.Image):
        """
        Prepare the mask tensor for the inpainting process.
        :param mask_image: The mask image
        :return: The normalized and resized mask tensor
        """
        mask_np = np.array(mask_image).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        latent_height = mask_tensor.shape[2] // 8
        latent_width = mask_tensor.shape[3] // 8
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(latent_height, latent_width), mode='nearest')
        return mask_tensor

    def decode_latents(self, latents) -> Image.Image:
        """
        Decode the latents into an image using the VAE.
        :param latents: The latents to decode.
        :return: The decoded image
        (This function was implemented with the help of AI)
        """
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)

    @torch.no_grad()
    def _initialize_denoise_loop(self, init_latents, mask_tensor, num_inference_steps: int):
        """
        Initialize the denoising loop.
        :param init_latents: The initial latents.
        :param mask_tensor: The mask tensor.
        :param num_inference_steps: The number of inference steps.
        :return: The modified initial latents and scheduler timesteps.
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        noise = torch.randn_like(init_latents)

        timesteps = self.scheduler.timesteps
        if self.reconstruction:
            latent_noise_scale = 0.2
            textured_latents = init_latents + (torch.randn_like(init_latents) * latent_noise_scale * mask_tensor)
            init_step = min(int(num_inference_steps * (1 - self.init_noise_strength)), num_inference_steps - 1)
            timesteps = self.scheduler.timesteps[init_step:]
            latents = self.scheduler.add_noise(textured_latents, noise, timesteps[0])
        else:
            latents = ((self.scheduler.add_noise(init_latents, noise, timesteps[0]) * (1 - mask_tensor))
                       + (noise * mask_tensor))
        return latents, timesteps

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps: int = 50):
        """
        Denoise the initial latents using the UNet - generate the masked area DDPM style.
        :param text_embeddings: The text embeddings.
        :param init_latents: The initial latents.
        :param mask_tensor: The mask tensor.
        :param num_inference_steps: The number of inference steps.
        :return: The denoised latents.
        """
        latents, timesteps = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                noise = torch.randn_like(init_latents)
                known_background = self.scheduler.add_noise(init_latents, noise, t_next)
            else:
                known_background = init_latents

            latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        return latents

    def mask_preprocessing(self, mask_image: Image.Image) -> Image.Image:
        """
        Preprocess the mask image.
        :param mask_image: The mask image to preprocess.
        :return: The preprocessed mask image.
        """
        return mask_image

    def image_preprocessing(self, real_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        """
        Preprocess the real image.
        :param real_image: The real image to preprocess.
        :param mask_image: The mask to be used to preprocess the real image with.
        :return: The preprocessed real image.
        """
        return real_image

    def preprocess(self, pipe_in: InpaintPipelineInput):
        """
        Preprocess the input data.
        :param pipe_in: The input data.
        :return: The preprocessed input data.
        """
        pipe_in.mask_image = self.mask_preprocessing(pipe_in.mask_image)
        pipe_in.init_image = self.image_preprocessing(pipe_in.init_image, pipe_in.mask_image)
        return pipe_in

    @torch.no_grad()
    def pipe(self, pipe_in: InpaintPipelineInput, target_size=512):
        """
        The main pipeline function.
        :param pipe_in: The input data.
        :param target_size: The target size of the model's input image.
        """
        pipe_in = self.preprocess(pipe_in)
        orig_size = pipe_in.init_image.size
        pipe_in.init_image, pipe_in.mask_image = self.prepare_inpainting_data_advanced(
            pipe_in.init_image, pipe_in.mask_image, target_size
        )
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
        return self.restore_original_dimensions(final_image, orig_size)
