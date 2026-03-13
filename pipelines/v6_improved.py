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
        (This function was implemented with the help of AI)
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
        (This function was implemented with the help of AI)
        """
        processor_dict = {
            name: AttnProcessor2_0() for name in self.unet.attn_processors.keys()
        }
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
