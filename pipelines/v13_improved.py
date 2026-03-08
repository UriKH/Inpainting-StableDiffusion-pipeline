from pipelines.v10_improved import ImprovedInpaintPipelineV10
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F


class ImprovedInpaintPipelineV13(ImprovedInpaintPipelineV10):
    def __init__(self, dmb_dilation_kernel_size=3, dmb_blur_kernel_size=5, dmb_sigma=5, **kwargs):
        super().__init__(**kwargs)
        self.dmb_dilation_kernel_size = dmb_dilation_kernel_size
        self.dmb_blur_kernel_size = dmb_blur_kernel_size
        self.dmb_sigma = dmb_sigma

    def _get_dynamic_mask(self, mask_tensor, current_idx, total_steps):
        """Linearly anneals the mask blur from max_sigma down to a hard boundary."""
        progress = current_idx / max(1, (total_steps - 1))
        # Decay sigma as we get closer to step 0
        current_sigma = max(0.01, self.dmb_sigma * (1.0 - progress))

        if current_sigma <= 0.5:
            return mask_tensor  # Return hard mask at the end for perfect background

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
        """Overrides the base denoise method to include time-travel resampling."""
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 1. Generate the custom time-travel schedule
        schedule_indices = self._get_repaint_schedule(num_inference_steps)
        
        # 2. Initial Setup
        noise = torch.randn_like(init_latents)
        latents = ((self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[schedule_indices[0]]) * (1 - mask_tensor))
                   + (noise * mask_tensor))
        
        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = self.scheduler.timesteps[step_index]
    
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                # Predict noise
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.CFG_SCALE_FACTOR * (noise_pred_text - noise_pred_uncond)
    
                # Step the scheduler to get t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # 3. Handle Time-Travel and Background Context
                is_last_step = (idx == len(schedule_indices) - 1)
            
                if not is_last_step:
                    next_step_index = schedule_indices[idx + 1]
                    t_next = self.scheduler.timesteps[next_step_index]
                
                    is_jump_backward = next_step_index < step_index
                
                    if is_jump_backward:
                        # Time travel! Calculate the precise DDPM transition ratio
                        # Get the alpha for where we currently are (step_index + 1)
                        if step_index + 1 < num_inference_steps:
                            t_prev = self.scheduler.timesteps[step_index + 1]
                            alpha_prod_prev = self.scheduler.alphas_cumprod[t_prev].to(self.device)
                        else:
                            # If we just finished the final step, current alpha is 1.0 (clean)
                            alpha_prod_prev = torch.tensor(1.0, device=self.device)
                            
                        # Get the alpha for where we are jumping to
                        alpha_prod_target = self.scheduler.alphas_cumprod[t_next].to(self.device)
                        
                        # Calculate the ratio and precisely inject the block-noise
                        ratio = alpha_prod_target / alpha_prod_prev
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(ratio) * latents + torch.sqrt(1 - ratio) * noise
                    
                    # Generate fresh stochastic background context for the upcoming step
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    # Final cleanup step resolving to sharpness
                    known_background = init_latents
    
                # 4. Blend the accurately aligned latents
                dynamic_mask = self._get_dynamic_mask(mask_tensor, idx, len(schedule_indices))
                latents = (known_background * (1 - dynamic_mask)) + (latents * dynamic_mask)
        finally:
            self._remove_masked_attention()
        return latents
