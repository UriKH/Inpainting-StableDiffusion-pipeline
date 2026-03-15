from pipelines.v6_improved import ImprovedInpaintPipelineV6
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from pipelines.injector import Injector


class ImprovedInpaintPipelineV7(ImprovedInpaintPipelineV6):
    def __init__(self, sm_dilation_kernel=3, sm_blur_kernel=5, sm_sigma=5.0, **kwargs):
        """
        :param sm_dilation_kernel: The size of the kernel for dilation for soft masking.
        :param sm_blur_kernel: The size of the kernel for Gaussian blur for soft masking.
        :param sm_sigma: The standard deviation of the Gaussian blur for soft masking.
        """
        super().__init__(**kwargs)

        self.dilation_kernel = sm_dilation_kernel
        self.blur_kernel = sm_blur_kernel
        self.sigma = sm_sigma

    def __create_soft_mask(self, mask_tensor):
        """
        Applies dilation and Gaussian blur to the binary mask tensor.
        :param mask_tensor: The binary mask tensor.
        :return: The softened mask tensor (dilated and feathered).
        """
        pad = self.dilation_kernel // 2
        dilated_mask = F.max_pool2d(mask_tensor, kernel_size=self.dilation_kernel, stride=1, padding=pad)
        soft_mask = TF.gaussian_blur(dilated_mask, kernel_size=[self.blur_kernel, self.blur_kernel], sigma=[self.sigma, self.sigma])
        return torch.clamp(soft_mask, 0.0, 1.0)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps=50):
        latents, timesteps = self.__initialize_denoise_loop(init_latents, mask, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape

        soft_attn_mask = self.__create_soft_mask(mask)
        self.unet = Injector.inject(
            unet=self.unet,
            latent_h=latent_h,
            latent_w=latent_w,
            self_mask=None,
            cross_mask=soft_attn_mask,
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
