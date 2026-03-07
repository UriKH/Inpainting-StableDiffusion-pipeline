from pipelines.v5_improved import ImprovedInpaintPipelineV5
import torch


class ImprovedInpaintPipelineV6(ImprovedInpaintPipelineV5):
    def __init__(self, rp_jump_length=10, rp_jump_n_sample=2, **kwargs):
        super().__init__(**kwargs)
        self.jump_length = rp_jump_length
        self.jump_n_sample = rp_jump_n_sample
        print(self.jump_length)
    
    def _get_repaint_schedule(self, num_inference_steps):
        """Generates the RePaint sequence of timestep indices."""
        times = list(range(num_inference_steps))
        schedule_indices = []

        i = 0
        jumps_done = 0
        while i < len(times):
            schedule_indices.append(i)
            # Check if we need to jump back in time
            if (i + 1) % self.jump_length == 0 and jumps_done < self.jump_n_sample - 1:
                i = i - self.jump_length + 1
                jumps_done += 1
            else:
                # Reset jump counter and move forward
                if (i + 1) % self.jump_length == 0:
                    jumps_done = 0
                i += 1

        return schedule_indices

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
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor)
        
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
                        # Time travel! Apply the DDPM forward equation to inject noise back in
                        beta = self.scheduler.betas[t_next].to(self.device)
                        noise = torch.randn_like(latents)
                        latents = torch.sqrt(1 - beta) * latents + torch.sqrt(beta) * noise
                    
                    # Generate fresh stochastic background context for the upcoming step
                    background_noise = torch.randn_like(init_latents)
                    known_background = self.scheduler.add_noise(init_latents, background_noise, t_next)
                else:
                    # Final cleanup step resolving to sharpness
                    known_background = init_latents
    
                # 4. Blend the accurately aligned latents
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents
