from pipelines.v11_improved import ImprovedInpaintPipelineV11
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils.torch_utils import generate_perlin_noise_2d


class ImprovedInpaintPipelineV12(ImprovedInpaintPipelineV11):
    def __init__(self, om_noise_res=16, om_dilation_kernel=5, om_thresh='linear', om_blur_kernel=1, **kwargs):
        """
        :param om_noise_res: The resolution of the noise map for organic masking.
        :param om_dilation_kernel: The size of the kernel for dilation for organic masking.
        :param om_thresh: The threshold type for organic masking.
        :param om_blur_kernel: The size of the kernel for Gaussian blur for organic masking.
        """
        super().__init__(**kwargs)
        self.om_noise_res = om_noise_res
        self.om_dilation_kernel = om_dilation_kernel
        self.om_thresh = om_thresh
        self.om_blur_kernel = om_blur_kernel

    def compute_organic_mask(self, base_mask, t, max_t):
        """
        Generates a time-dependent, organic 'coastline' mask for latent blending.
        :param base_mask: The base mask tensor.
        :param t: The current timestep.
        :param max_t: The maximum number of timesteps.
        """
        if t.item() == 0:
            return base_mask

        rho = 1.0 - (t.item() / max_t)
        pad = self.om_dilation_kernel // 2
        dilated_mask = F.max_pool2d(base_mask, kernel_size=self.om_dilation_kernel, stride=1, padding=pad)
        smooth_noise = generate_perlin_noise_2d(
            shape=base_mask.shape,
            res=(self.om_noise_res, self.om_noise_res),
            device=base_mask.device
        )

        noisy_mask = dilated_mask * smooth_noise
        if self.om_thresh == 'linear':
            current_threshold = 0.2 + (rho * 0.6)
        elif self.om_thresh == 'cubic':
            current_threshold = 0.2 + ((rho ** 2) * 0.6)
        else:
            raise ValueError(f'bad threshold type {self.om_thresh}. Use: linear/cubic')

        dynamic_mask = (noisy_mask > current_threshold).float()
        extended = torch.max(dynamic_mask, base_mask)
        if self.om_blur_kernel != 1:
            extended = TF.gaussian_blur(
                extended,
                kernel_size=[self.om_blur_kernel, self.om_blur_kernel]
            )
        return extended

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps: int = 50):
        latents, timesteps, schedule_indices = self.__initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)

        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask,
                                      mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        max_t = self.scheduler.config.num_train_timesteps

        try:
            for idx, step_index in enumerate(schedule_indices):
                t = timesteps[step_index]

                # create organic mask
                organic_mask = self.compute_organic_mask(
                    base_mask=mask_tensor,
                    t=t,
                    max_t=max_t
                )

                # inject organic mask
                for name, proc in self.unet.attn_processors.items():
                    if 'attn1' in name:
                        proc.mask_tensor = organic_mask
                    elif 'attn2' in name:
                        proc.mask_tensor = self._create_soft_mask(organic_mask)

                latents = self.__denoise_step(t, text_embeddings, latents)

                if idx != len(schedule_indices) - 1:
                    scheduler_next_step = schedule_indices[idx + 1]
                    t_next = timesteps[scheduler_next_step]

                    if scheduler_next_step < step_index:
                        latents = self.__resampling_latent_update(latents, t_next, step_index, timesteps)

                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    known_background = init_latents

                latents = (known_background * (1 - organic_mask)) + (latents * organic_mask)
        finally:
            self._remove_masked_attention()
        return latents
