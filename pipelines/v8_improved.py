from pipelines.self_attention import MaskedSelfAttnProcessor
from pipelines.v7_improved import ImprovedInpaintPipelineV7
import torch
from diffusers.models.attention_processor import AttnProcessor2_0


class ImprovedInpaintPipelineV8(ImprovedInpaintPipelineV7):
    def __init__(self, sa_dilation_threshold: float = 0.0, sa_resize_mode: str = 'nearest', **kwargs):
        """
        :param sa_dilation_threshold: The threshold for dilation in the self-attention layer.
        :param sa_resize_mode: The resize mode for the self-attention layer.
        """
        super().__init__(**kwargs)
        self.sa_dilation_threshold = sa_dilation_threshold
        self.sa_resize_mode = sa_resize_mode

    def _inject_masked_attention(self, latent_h, latent_w, self_mask):
        """
        Injects custom Self Attention processors into the UNet.
        :param latent_h: The height of the latent space.
        :param latent_w: The width of the latent space.
        :param self_mask: The binary mask tensor.
        (This function was implemented with the help of AI)
        """
        processor_dict = {}
        for name in self.unet.attn_processors.keys():
            if "attn1" in name:
                processor = MaskedSelfAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = self_mask
                processor.dilation_threshold = self.sa_dilation_threshold
                processor.resize_mode = self.sa_resize_mode
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0()
        self.unet.set_attn_processor(processor_dict)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        latents, timesteps = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)

        _, _, latent_h, latent_w = init_latents.shape
        self._inject_masked_attention(latent_h, latent_w, mask_tensor)

        try:
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
        finally:
            self._remove_masked_attention()
        return latents
