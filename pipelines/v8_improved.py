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
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps=50):
        latents, timesteps = self.__initialize_denoise_loop(init_latents, mask, num_inference_steps)

        _, _, latent_h, latent_w = init_latents.shape
        self._inject_masked_attention(latent_h, latent_w, mask)

        try:
            for i, t in enumerate(timesteps):
                latents = self.__denoise_step(t, text_embeddings, latents)

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    noise = torch.randn_like(init_latents)
                    background = self.scheduler.add_noise(init_latents, noise, t_next)
                else:
                    background = init_latents
                latents = (background * (1 - mask)) + (latents * mask)
        finally:
            self._remove_masked_attention()
        return latents
