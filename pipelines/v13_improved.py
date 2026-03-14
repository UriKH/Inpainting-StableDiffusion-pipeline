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

    def _get_dynamic_mask(self, mask_tensor, step_index, total_diffusion_steps):
        """
        Linearly anneals the mask blur from max_sigma down to a hard boundary.
        :param mask_tensor: The binary mask tensor.
        :param step_index: The current step index.
        :param total_diffusion_steps: The total number of steps.
        :return: The dynamic mask tensor.
        """
        progress = step_index / max(1, (total_diffusion_steps - 1))
        current_sigma = max(0.01, self.dmb_sigma * (1.0 - progress))

        if current_sigma <= 0.1:
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
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        latents, timesteps, schedule_indices = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)
        _, _, latent_h, latent_w = init_latents.shape

        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = timesteps[step_index]
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if idx != len(schedule_indices) - 1:
                    next_step_index = schedule_indices[idx + 1]
                    t_next = timesteps[next_step_index]
                
                    is_jump_backward = next_step_index < step_index
                
                    if is_jump_backward:
                        if step_index + 1 < len(timesteps):
                            t_prev = timesteps[step_index + 1]
                            alpha_prod_prev = self.scheduler.alphas_cumprod[t_prev].to(self.device)
                        else:
                            alpha_prod_prev = torch.tensor(1.0, device=self.device)
                            
                        alpha_prod_target = self.scheduler.alphas_cumprod[t_next].to(self.device)
                        ratio = alpha_prod_target / alpha_prod_prev
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(ratio) * latents + torch.sqrt(1 - ratio) * noise
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    known_background = init_latents

                dynamic_mask = self._get_dynamic_mask(mask_tensor, step_index, len(timesteps))
                latents = (known_background * (1 - dynamic_mask)) + (latents * dynamic_mask)
        finally:
            self._remove_masked_attention()
        return latents
