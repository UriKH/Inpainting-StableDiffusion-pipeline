from code.pipelines.v8_improved import ImprovedInpaintPipelineV8
import torch
from code.pipelines.injector import Injector


class ImprovedInpaintPipelineV9(ImprovedInpaintPipelineV8):
    def __init__(self, use_sm_in_sa=False, **kwargs):
        super().__init__(**kwargs)
        self.use_sm_in_sa = use_sm_in_sa

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps=50):
        latents, timesteps = self.__initialize_denoise_loop(init_latents, mask, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape

        soft_attn_mask = self.__create_soft_mask(mask)
        self.unet = Injector.inject(
            unet=self.unet,
            latent_h=latent_h,
            latent_w=latent_w,
            self_mask=mask if not self.use_sm_in_sa else soft_attn_mask,
            cross_mask=soft_attn_mask,
            ignore_cross_attention=self.ignore_cross_attention,
            ca_resize_mode=self.ca_resize_mode,
            sa_resize_mode=self.sa_resize_mode,
            sa_dilation_threshold=self.sa_dilation_threshold
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
