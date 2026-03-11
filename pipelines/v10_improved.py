from pipelines.v9_improved import ImprovedInpaintPipelineV9
import torch
import math


class ImprovedInpaintPipelineV10(ImprovedInpaintPipelineV9):
    def __init__(self, rp_jump_length=10, rp_jump_n_sample=2,
                 ds_min_jumps=1, ds_min_jump_len=5, ds_max_jumps=4, ds_max_jump_len=10,
                 use_dynamic_schedule=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.jump_length = rp_jump_length
        self.jump_n_sample = rp_jump_n_sample
        self.ds_min_jumps = ds_min_jumps
        self.ds_min_jump_len = ds_min_jump_len
        self.ds_max_jumps = ds_max_jumps
        self.ds_max_jump_len = ds_max_jump_len
        self.use_dynamic_schedule = use_dynamic_schedule
    
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

    def _get_dynamic_schedule(self, num_inference_steps):
        """Generates the dynamic sequence of timestep indices."""
        times = list(range(num_inference_steps))
        schedule_indices = []
        i = 0

        while i < len(times):
            progress = i / len(times)
            curr_jump_length = int(math.cos(progress * math.pi / 2.) * (self.ds_max_jump_len - self.ds_min_jump_len) +  self.ds_min_jump_len)
            curr_jumps = int(math.cos(progress * math.pi / 2.) * (self.ds_max_jumps - self.ds_min_jumps) + self.ds_min_jumps)

            slice_len = min(curr_jump_length, len(times) - i)
            schedule_indices.extend([i + v for v in range(slice_len)] * curr_jumps)
            i += curr_jump_length
        return schedule_indices

    @torch.no_grad()
    def _initialize_denoise_loop(self, init_latents, mask_tensor, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        noise = torch.randn_like(init_latents)

        timesteps = self.scheduler.timesteps
        if self.reconstruction:
            init_step = min(int(num_inference_steps * (1 - self.init_noise_strength)), num_inference_steps - 1)
            timesteps = self.scheduler.timesteps[init_step:]

        if self.use_dynamic_schedule:
            schedule_indices = self._get_dynamic_schedule(len(timesteps))
        else:
            schedule_indices = self._get_repaint_schedule(len(timesteps))

        if self.reconstruction:
            latents = self.scheduler.add_noise(init_latents, noise, timesteps[schedule_indices[0]])
        else:
            latents = ((self.scheduler.add_noise(init_latents, noise, timesteps[schedule_indices[0]])
                        * (1 - mask_tensor)) + (noise * mask_tensor))
        return latents, timesteps, schedule_indices

    @torch.no_grad()
    def denoise(self, text_embeddings, init_latents, mask_tensor, num_inference_steps=50):
        """Overrides the base denoise method to include time-travel resampling."""
        latents, timesteps, schedule_indices = self._initialize_denoise_loop(init_latents, mask_tensor, num_inference_steps)
        
        _, _, latent_h, latent_w = init_latents.shape
        soft_attn_mask = self._create_soft_mask(mask_tensor)
        self._inject_masked_attention(latent_h, latent_w, soft_attn_mask, mask_tensor if not self.use_sm_in_sa else soft_attn_mask)
        
        try:
            for idx, step_index in enumerate(schedule_indices):
                t = timesteps[step_index]
    
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
                latents = (known_background * (1 - mask_tensor)) + (latents * mask_tensor)
        finally:
            self._remove_masked_attention()
        return latents
