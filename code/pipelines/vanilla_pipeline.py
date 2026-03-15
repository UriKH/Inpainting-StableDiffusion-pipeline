from pipelines.pipeline import InpaintingPipeLineScheme, InpaintPipelineInput
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import torch_utils as utils
from utils.globals import SD2_BASE
import cv2 as cv
from pipelines.vae_prepare import VaeConverter


class InpaintPipelineVanilla(InpaintingPipeLineScheme):
    MODEL_ID = SD2_BASE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CFG_SCALE_FACTOR = 7.5

    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE, **kwargs):
        super().__init__(model_id, device, **kwargs)

    def encode_prompt(self, prompt: str, encoder, tokenizer):
        """
        Create the prompt embeddings and the unconditional prompt embeddings.
        :param prompt: The prompt to encode
        :param encoder: The text encoder model
        :param tokenizer: The tokenizer model
        :return: The prompt embeddings [uncond, text]
        """
        text_input = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        text_embeddings = encoder(text_input.input_ids.to(self.device))[0]
        uncond_embeddings = encoder(uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def prepare_latents(self, image: Image.Image):
        """
        Prepare the latent variables for the inpainting process.
        :param image: The image to prepare the latents for
        :return: The prepared latents
        """
        image_tensor = VaeConverter.pil_to_tensor(image, self.device)
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

    @torch.no_grad()
    def decode_latents(self, latents) -> Image.Image:
        """
        Decode the latents into an image using the VAE.
        :param latents: The latents to decode.
        :return: The decoded image
        """
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        return VaeConverter.tensor_to_pil(image)

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
            init_step = min(int(num_inference_steps * (1 - self.init_noise_strength)), num_inference_steps - 1)
            timesteps = self.scheduler.timesteps[init_step:]
            latents = self.scheduler.add_noise(init_latents, noise, timesteps[0])
        else:
            latents = ((self.scheduler.add_noise(init_latents, noise, timesteps[0]) * (1 - mask_tensor))
                       + (noise * mask_tensor))
        return latents, timesteps

    @torch.no_grad()
    def __denoise_step(self, t, text_embeddings, latents):
        """
        Preform scaling, a single denoising step and guidance.
        :param t: The current timestep.
        :param text_embeddings: The text embeddings.
        :param latents: The current latents.
        """
        latent_in = torch.cat([latents] * 2)
        latent_in = self.scheduler.scale_model_input(latent_in, t)
        pred_noise = self.unet(latent_in, t, encoder_hidden_states=text_embeddings).sample
        unc_pred_noise, text_pred_noise = pred_noise.chunk(2)
        pred_noise = self.CFG_SCALE_FACTOR * text_pred_noise + (1 - self.CFG_SCALE_FACTOR) * unc_pred_noise
        latents = self.scheduler.step(pred_noise, t, latents).prev_sample
        return latents

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps: int = 50):
        """
        Denoise the initial latents using the UNet - generate the masked area DDPM style.
        :param text_embeddings: The text embeddings.
        :param init_latents: The initial latents.
        :param mask: The mask tensor.
        :param num_inference_steps: The number of inference steps.
        :return: The denoised latents.
        """
        latents, timesteps = self._initialize_denoise_loop(init_latents, mask, num_inference_steps)

        for i, t in enumerate(timesteps):
            latents = self.__denoise_step(t, text_embeddings, latents)

            # add noise to latents only if not finished after this step
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                noise = torch.randn_like(init_latents)
                background = self.scheduler.add_noise(init_latents, noise, t_next)
            else:
                background = init_latents

            latents = (background * (1 - mask)) + (latents * mask)
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
