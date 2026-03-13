from pipelines.cross_attention import MaskedCrossAttnProcessor
from pipelines.self_attention import MaskedSelfAttnProcessor
from pipelines.v8_improved import ImprovedInpaintPipelineV8
from diffusers.models.attention_processor import AttnProcessor2_0
import torch


class ImprovedInpaintPipelineV9(ImprovedInpaintPipelineV8):
    def __init__(self, use_sm_in_sa=False, **kwargs):
        super().__init__(**kwargs)
        self.use_sm_in_sa = use_sm_in_sa

    def _inject_masked_attention(self, latent_h, latent_w, cross_mask, self_mask):
        """
        Injects the custom Cross and Self Attention processors into the UNet.
        :param latent_h: The height of the latent space.
        :param latent_w: The width of the latent space.
        :param cross_mask: The cross-attention mask tensor.
        :param self_mask: The self-attention mask tensor.
        """
        processor_dict = {}
        for name in self.unet.attn_processors.keys():
            if "attn1" in name:  # Self-Attention Layers
                processor = MaskedSelfAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = self_mask
                processor.dilation_threshold = self.sa_dilation_threshold
                processor.resize_mode = self.sa_resize_mode
                processor_dict[name] = processor
            elif "attn2" in name and not self.ignore_cross_attention:  # Cross-Attention Layers
                processor = MaskedCrossAttnProcessor(latent_h, latent_w)
                processor.mask_tensor = cross_mask
                processor.resize_mode = self.ca_resize_mode
                processor_dict[name] = processor
            else:
                processor_dict[name] = AttnProcessor2_0()
        self.unet.set_attn_processor(processor_dict)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        latents, timesteps = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape

        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        
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
