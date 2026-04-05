from pipelines.v7_improved import ImprovedInpaintPipelineV7
import torch
from pipelines.injector import Injector


class ImprovedInpaintPipelineV8(ImprovedInpaintPipelineV7):
    def __init__(self, sa_dilation_threshold: float = 0.0, sa_resize_mode: str = 'nearest', **kwargs):
        """
        :param sa_dilation_threshold: The threshold for dilation in the self-attention layer.
        :param sa_resize_mode: The resize mode for the self-attention layer.
        """
        super().__init__(**kwargs)
        self.sa_dilation_threshold = sa_dilation_threshold
        self.sa_resize_mode = sa_resize_mode

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps=50):
        latents, timesteps = self._initialize_denoise_loop(init_latents, mask, num_inference_steps)

        _, _, latent_h, latent_w = init_latents.shape
        self.unet = Injector.inject(
            unet=self.unet,
            latent_h=latent_h,
            latent_w=latent_w,
            self_mask=mask,
            cross_mask=None,
            ignore_cross_attention=self.ignore_cross_attention,
            ca_resize_mode=self.ca_resize_mode,
            sa_resize_mode=self.sa_resize_mode,
            sa_dilation_threshold=self.sa_dilation_threshold
        )

        try:
            for i, t in enumerate(timesteps):
                latents = self._denoise_step(t, text_embeddings, latents)

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    noise = torch.randn_like(init_latents)
                    background = self.scheduler.add_noise(init_latents, noise, t_next)
                else:
                    background = init_latents
                latents = (background * (1 - mask)) + (latents * mask)
        finally:
            self.unet = Injector.remove(self.unet)
        return latents
