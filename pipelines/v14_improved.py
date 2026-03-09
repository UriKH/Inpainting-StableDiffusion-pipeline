from pipelines.v12_improved import ImprovedInpaintPipelineV12
from pipelines.v13_improved import ImprovedInpaintPipelineV13
import torch


class ImprovedInpaintPipelineV14(ImprovedInpaintPipelineV13, ImprovedInpaintPipelineV12):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        """Overrides the base denoise method to include time-travel resampling."""
        latents, timesteps, schedule_indices = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)

        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask,
                                      mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        max_t = self.scheduler.config.num_train_timesteps

        try:
            for idx, step_index in enumerate(schedule_indices):
                t = timesteps[step_index]

                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # update injection
                organic_mask = self.compute_organic_mask(
                    base_mask=mask_tensor,
                    t=t,
                    max_t=max_t
                )

                dynamic_soft_mask = self._create_soft_mask(organic_mask)
                for name, proc in self.unet.attn_processors.items():
                    if 'attn1' in name:
                        proc.mask_tensor = organic_mask
                    elif 'attn2' in name:
                        proc.mask_tensor = dynamic_soft_mask

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
                    t_next = timesteps[next_step_index]

                    is_jump_backward = next_step_index < step_index

                    if is_jump_backward:
                        # Time travel! Calculate the precise DDPM transition ratio
                        # Get the alpha for where we currently are (step_index + 1)
                        if step_index + 1 < len(timesteps):
                            t_prev = timesteps[step_index + 1]
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
                dynamic_mask = self._get_dynamic_mask(organic_mask, idx, len(schedule_indices))
                latents = (known_background * (1 - dynamic_mask)) + (latents * dynamic_mask)
        finally:
            self._remove_masked_attention()
        return latents
