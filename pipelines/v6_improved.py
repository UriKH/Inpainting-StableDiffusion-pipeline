from pipelines.cross_attention import MaskedCrossAttnProcessor
from pipelines.v5_improved import ImprovedInpaintPipelineV5
import torch
from diffusers.models.attention_processor import AttnProcessor2_0


class ImprovedInpaintPipelineV6(ImprovedInpaintPipelineV5):
    def __init__(self, ca_resize_mode = 'area', ignore_cross_attention=False, **kwargs):
        """
        :param ca_resize_mode: The resize mode for cross-attention.
        :param ignore_cross_attention: Whether to ignore cross-attention upgrade.
        """
        super().__init__(**kwargs)
        self.ca_resize_mode = ca_resize_mode
        self.ignore_cross_attention = ignore_cross_attention

    def _inject_masked_attention(self, latent_h, latent_w, mask_tensor):
        """
        Replaces standard cross-attention with the Masked processor.
        (This function was implemented with the assistance of AI)
        """
        processor_dict = {}
        for name in self.unet.attn_processors.keys():
            if "attn2" in name and not self.ignore_cross_attention:
                processor = MaskedCrossAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = mask_tensor
                processor.resize_mode = self.ca_resize_mode
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0()
        self.unet.set_attn_processor(processor_dict)

    def _remove_masked_attention(self):
        """
        Restores the UNet to its vanilla state to prevent side effects.
        (This function was implemented with the assistance of AI)
        """
        processor_dict = {
            name: AttnProcessor2_0() for name in self.unet.attn_processors.keys()
        }
        self.unet.set_attn_processor(processor_dict)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps: int = 50):
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
