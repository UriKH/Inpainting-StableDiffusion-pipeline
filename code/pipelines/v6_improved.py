from code.pipelines.v5_improved import ImprovedInpaintPipelineV5
import torch
from code.pipelines.injector import Injector


class ImprovedInpaintPipelineV6(ImprovedInpaintPipelineV5):
    def __init__(self, ca_resize_mode = 'area', ignore_cross_attention=False, **kwargs):
        """
        :param ca_resize_mode: The resize mode for cross-attention.
        :param ignore_cross_attention: Whether to ignore cross-attention upgrade.
        """
        super().__init__(**kwargs)
        self.ca_resize_mode = ca_resize_mode
        self.ignore_cross_attention = ignore_cross_attention

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps: int = 50):
        latents, timesteps = self.__initialize_denoise_loop(init_latents, mask, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape
        self.unet = Injector.inject(
            unet=self.unet,
            latent_h=latent_h,
            latent_w=latent_w,
            self_mask=None,
            cross_mask=mask,
            ignore_cross_attention=self.ignore_cross_attention,
            ca_resize_mode=self.ca_resize_mode,
            sa_resize_mode=None,
            sa_dilation_threshold=None
        )

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
            self.unet = Injector.remove(self.unet)
        return latents
