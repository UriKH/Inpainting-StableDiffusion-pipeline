from pipelines.injector import Injector
from pipelines.v11_improved import ImprovedInpaintPipelineV11
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F


class ImprovedInpaintPipelineV13(ImprovedInpaintPipelineV11):
    def __init__(self, dmb_dilation_kernel_size=3, dmb_blur_kernel_size=5, dmb_sigma=5, **kwargs):
        """
        :param dmb_dilation_kernel_size: The size of the kernel for dilation for dynamic mask blurring.
        :param dmb_blur_kernel_size: The size of the kernel for Gaussian blur for dynamic mask blurring.
        :param dmb_sigma: The standard deviation of the Gaussian blur for dynamic mask blurring.
        """
        super().__init__(**kwargs)
        self.dmb_dilation_kernel_size = dmb_dilation_kernel_size
        self.dmb_blur_kernel_size = dmb_blur_kernel_size
        self.dmb_sigma = dmb_sigma

    def __get_dynamic_mask(self, mask_tensor, current_idx, total_steps):
        """
        Linearly anneals the mask blur from max_sigma down to a hard boundary.
        :param mask_tensor: The binary mask tensor.
        :param current_idx: The current step index.
        :param total_steps: The total number of steps.
        :return: The dynamic mask tensor.
        """
        progress = current_idx / max(1, (total_steps - 1))
        current_sigma = max(0.01, self.dmb_sigma * (1.0 - progress))

        if current_sigma <= 0.5:
            return mask_tensor

        if self.dmb_dilation_kernel_size == 1:
            dilated_mask = mask_tensor
        else:
            padding = self.dmb_dilation_kernel_size // 2
            dilated_mask = F.max_pool2d(
                mask_tensor,
                kernel_size=self.dmb_dilation_kernel_size,
                stride=1,
                padding=padding
            )
        return TF.gaussian_blur(
            dilated_mask,
            kernel_size=[self.dmb_blur_kernel_size, self.dmb_blur_kernel_size],
            sigma=[current_sigma, current_sigma]
        )

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask, num_inference_steps=50):
        latents, timesteps, schedule_indices = self._initialize_denoise_loop(init_latents, mask, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape

        soft_attn_mask = self._create_soft_mask(mask)
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
            for i, step_index in enumerate(schedule_indices):
                t = timesteps[step_index]
                latents = self._denoise_step(t, text_embeddings, latents)

                if i != len(schedule_indices) - 1:
                    scheduler_next_step = schedule_indices[i + 1]
                    t_next = timesteps[scheduler_next_step]
                
                    if scheduler_next_step < step_index:
                        latents = self._resampling_latent_update(latents, t_next, step_index, timesteps)

                    background = self.scheduler.add_noise(init_latents, torch.randn_like(init_latents), t_next)
                else:
                    background = init_latents

                dynamic_mask = self.__get_dynamic_mask(mask, i, len(schedule_indices))
                latents = (background * (1 - dynamic_mask)) + (latents * dynamic_mask)
        finally:
            self.unet = Injector.remove(self.unet)
        return latents
